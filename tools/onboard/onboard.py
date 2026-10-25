#!/usr/bin/env python3
"""The deck onboarding pipeline (docs/open_vocab_plan.md S7; tools/onboard/README.md).

    python3 tools/onboard/onboard.py <stage|all|report|check> [--deck STEM ...] [options]

Stages, in order (a stage never runs past an earlier one that is not ok):

     1 scrape      top-N Legacy decks from MTGTop8 -> decks/candidates/<stem>.dk     (network)
     2 diff        each card against the vocab: in vocab / script, not in vocab / no script
     3 provision   fetch the missing scripts and their tokens at the pinned Forge commit (network)
     4 parse       the engine's --parse-report over the deck's new cards
     5 capability  per card clean / needs mechanic X; per deck missing cards and mechanics;
                   writes decks/candidates/REPORT.md (mechanics ranked by decks they unlock)
     6 implement   ok once no new card needs a mechanic (the engine work itself is a developer's)
     7 vocab       append the deck's new cards (and their tokens) to src/card_vocab.h, make registry
     8 text        make text-table (the new cards' oracle-text rows; network, libtorch)
     9 checks      make check, then the fuzz campaign robomage_check --tier fuzz --matchups <deck>:*
    10 designate   git mv the deck to decks/league/ (--to league) or decks/heldout/ (--to heldout)
    11 baseline    scripted:hard piloting the deck (rm_driver.py) and rmtrain bench --decks (S8)

`all` runs stages 2-6 (the automatic ones: read-only but for provision's fetches) for every deck;
the others run by name. Every stage records its result in the deck's status file
decks/candidates/<stem>.status.json (stage -> result, time, input hashes, summary, detail) and is
skipped while its inputs are unchanged and its result ok (--force reruns it). Stages 2-6 rerun
themselves when a later stage needs them and their inputs changed; stages 7-10 need only an ok.
`report` prints every status file as one table; `check` reruns stages 2, 4 and 5 for every
candidate without writing anything or touching the network and reports any committed status file
whose result is out of date (the gate's vocab tier runs it). Standard library only.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BIN = os.path.join(REPO, "bin")
RESOURCES = os.path.join(BIN, "resources")
DECKS = os.path.join(RESOURCES, "decks")
CARDS = os.path.join(RESOURCES, "cardsfolder")
TOKENS = os.path.join(RESOURCES, "tokenscripts")
VOCAB_H = os.path.join(REPO, "src", "card_vocab.h")
TEXT_TABLE = os.path.join(RESOURCES, "embeddings", "oracle_text.rmtb")
DECK_DIRS = ("candidates", "league", "heldout")
STATUS_DIR = os.path.join(DECKS, "candidates")
REPORT_MD = os.path.join(STATUS_DIR, "REPORT.md")
SIZES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mechanic_sizes.json")
SCRAPER = os.path.join(REPO, "tools", "meta_scraper", "scrape_meta.py")
FETCHER = os.path.join(REPO, "tools", "forge_fetch", "fetch_script.py")
DRIVER = os.path.join(REPO, "python", "rm_driver.py")
RMTRAIN = os.path.join(REPO, "learn", "rmtrain")

STAGES = ["scrape", "diff", "provision", "parse", "capability", "implement", "vocab", "text", "checks",
          "designate", "baseline"]
AUTOMATIC = ["diff", "provision", "parse", "capability", "implement"]
TOKEN_BASE, TOKEN_LAST = 900, 1022

sys.path.insert(0, os.path.join(REPO, "tools", "forge_fetch"))
import provision_decks  # noqa: E402  (collect_tokens: the token scripts a set of cards makes)


# ── Small helpers ───────────────────────────────────────────────────────────────────────────────

def now():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    if not os.path.exists(path):
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def name_to_uid(name):
    """src/parse.cpp name_to_uid: lowercase, space/hyphen/slash -> '_', drop other characters,
    collapse runs of '_'."""
    uid = re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_").replace("/", "_"))
    return re.sub(r"_+", "_", uid)


def script_face_name(path, back):
    in_back = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line == "ALTERNATE":
                if not back:
                    return ""
                in_back = True
                continue
            if in_back == back and line.startswith("Name:"):
                return line[5:]
    return ""


def resolve_script(name):
    """The script load_card reads for `name` (src/card_db.cpp script_path_for), or None."""
    uid = name_to_uid(name)
    if not uid:
        return None
    letter_dir = os.path.join(CARDS, uid[0])
    exact = os.path.join(letter_dir, uid + ".txt")
    if os.path.exists(exact):
        return exact
    if os.path.isdir(letter_dir):
        for fn in sorted(os.listdir(letter_dir)):
            if fn.startswith(uid + "_") and fn.endswith(".txt") and len(fn) > len(uid) + 5:
                path = os.path.join(letter_dir, fn)
                if name_to_uid(script_face_name(path, False)) == uid:
                    return path
    for letter in sorted(os.listdir(CARDS)) if os.path.isdir(CARDS) else []:
        d = os.path.join(CARDS, letter)
        if letter.startswith(".") or not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith("_" + uid + ".txt"):
                path = os.path.join(d, fn)
                if name_to_uid(script_face_name(path, True)) == uid:
                    return path
    return None


def deck_path(stem):
    for d in DECK_DIRS:
        p = os.path.join(DECKS, d, stem + ".dk")
        if os.path.exists(p):
            return d, p
    return None, None


def deck_entries(path):
    """(quantity, name, section) per line of a .dk file (classes/deck.cpp's format)."""
    out, section = [], "main"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("SIDEBOARD"):
                section = "side"
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m:
                out.append((int(m.group(1)), m.group(2).strip(), section))
    return out


def deck_names(path):
    seen, out = set(), []
    for _, name, _ in deck_entries(path):
        if name_to_uid(name) not in seen:
            seen.add(name_to_uid(name))
            out.append(name)
    return out


def scripts_digest(names):
    """One hash over the script each name resolves to and its bytes (or its absence)."""
    lines = []
    for name in sorted(names, key=name_to_uid):
        p = resolve_script(name)
        lines.append("%s\t%s\t%s" % (name_to_uid(name), os.path.relpath(p, RESOURCES) if p else "-",
                                     sha256_file(p) if p else "-"))
    return sha256_bytes("\n".join(lines).encode())


def tokens_digest():
    names = sorted(os.listdir(TOKENS)) if os.path.isdir(TOKENS) else []
    return sha256_bytes("\n".join(names).encode())


class Engine:
    """The engine binary the stages run (cwd bin/, as the gate and the driver run it)."""

    def __init__(self, binary):
        self.binary = os.path.abspath(binary)
        self._digest = None

    def digest(self):
        if self._digest is None:
            self._digest = sha256_file(self.binary)
        return self._digest

    def run(self, args):
        if not os.path.exists(self.binary):
            raise RuntimeError("no engine at %s (make, or --binary)" % self.binary)
        return subprocess.run([self.binary] + args, cwd=BIN, capture_output=True, text=True)


def default_binary():
    config = "release" if os.environ.get("ROBOMAGE_BUILD") == "release" else "debug"
    return os.path.join(BIN, config, "robomage")


# ── Status files ────────────────────────────────────────────────────────────────────────────────

def status_path(stem):
    return os.path.join(STATUS_DIR, stem + ".status.json")


def load_status(stem):
    p = status_path(stem)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    loc, _ = deck_path(stem)
    return {"deck": stem, "location": loc, "source": {}, "stages": {}}


def save_status(status):
    loc, _ = deck_path(status["deck"])
    if loc:
        status["location"] = loc
    with open(status_path(status["deck"]), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def all_stems():
    stems = set()
    for fn in os.listdir(STATUS_DIR) if os.path.isdir(STATUS_DIR) else []:
        if fn.endswith(".status.json"):
            stems.add(fn[:-len(".status.json")])
        elif fn.endswith(".dk"):
            stems.add(fn[:-3])
    return sorted(stems)


def record(status, stage, result, inputs, summary, detail):
    status["stages"][stage] = {"result": result, "time": now(), "inputs": inputs, "summary": summary,
                               "detail": detail}


def stage_ok(status, stage):
    rec = status["stages"].get(stage)
    return rec is not None and rec["result"] == "ok"


# ── The stages ──────────────────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.engine = Engine(args.binary or default_binary())
        self.sizes = load_sizes()
        self.online = not getattr(args, "offline", False)
        self.write = True

    # Inputs: the hashes a stage's result depends on, computed fresh.
    def inputs(self, stage, status):
        _, path = deck_path(status["deck"])
        deck = sha256_file(path) if path else "none"
        if stage in ("diff", "provision"):
            return {"deck": deck, "vocab": sha256_file(VOCAB_H), "scripts": scripts_digest(deck_names(path))}
        if stage == "parse":
            return dict(self.inputs("diff", status), engine=self.engine.digest(), tokens=tokens_digest())
        if stage in ("capability", "implement"):
            return dict(self.inputs("parse", status), sizes=sha256_file(SIZES_JSON))
        if stage == "vocab":
            return {"deck": deck}
        if stage == "text":
            return {"vocab": sha256_file(VOCAB_H), "text_table": sha256_file(TEXT_TABLE)}
        if stage == "checks":
            return {"deck": deck, "vocab": sha256_file(VOCAB_H), "engine": self.engine.digest()}
        return {}

    def current(self, stage, status):
        rec = status["stages"].get(stage)
        return rec is not None and rec["result"] == "ok" and rec["inputs"] == self.inputs(stage, status)

    def ensure(self, stage, status, force=False):
        """Run `stage` for the deck unless it is current; first make every earlier stage ok
        (an automatic one rerun when stale). Returns the stage's result."""
        k = STAGES.index(stage)
        # A prerequisite can make an earlier one stale (provision fetches the scripts diff hashed),
        # so pass over them until all are current.
        for _ in range(len(AUTOMATIC) + 1):
            for prev in STAGES[1:k]:
                if prev in AUTOMATIC:
                    if self.ensure(prev, status) != "ok":
                        return "blocked"
                elif not stage_ok(status, prev):
                    print("  %s: %s needs %s first" % (status["deck"], stage, prev))
                    return "blocked"
            if all(self.current(p, status) for p in STAGES[1:k] if p in AUTOMATIC):
                break
        if not force and self.current(stage, status):
            return "ok"
        inputs = self.inputs(stage, status)
        result, summary, detail = getattr(self, "stage_" + stage)(status)
        if stage in ("provision", "vocab"):  # they change what their inputs hash
            inputs = self.inputs(stage, status)
        record(status, stage, result, inputs, summary, detail)
        if self.write:
            save_status(status)
        print("  %-24s %-10s %-4s %s" % (status["deck"], stage, result, summary))
        return result

    # 2 diff
    def stage_diff(self, status):
        loc, path = deck_path(status["deck"])
        r = self.engine.run(["--check-decks", "--check-decks-dir", "%s/%s.dk" % (loc, status["deck"])])
        if r.returncode not in (0, 1):
            return "fail", "--check-decks exited %d" % r.returncode, {"stderr": r.stderr[-2000:]}
        not_in_vocab, no_script, distinct = [], [], None
        for line in r.stdout.splitlines():
            m = re.match(r"^check-decks: MISSING (.+): (not in src/card_vocab.h|no script) \(", line)
            if m:
                (no_script if m.group(2) == "no script" else not_in_vocab).append(m.group(1))
            m = re.match(r"^check-decks: \d+ deck\(s\), (\d+) distinct card\(s\), (\d+) missing", line)
            if m:
                distinct = int(m.group(1))
        if distinct is None:
            return "fail", "no --check-decks summary", {"stdout": r.stdout[-2000:]}
        detail = {"distinct": distinct, "in_vocab": distinct - len(not_in_vocab) - len(no_script),
                  "not_in_vocab": sorted(not_in_vocab), "no_script": sorted(no_script)}
        return "ok", "%d cards: %d in vocab, %d new (%d without a script)" % (
            distinct, detail["in_vocab"], len(not_in_vocab) + len(no_script), len(no_script)), detail

    # 3 provision
    def stage_provision(self, status):
        diff = status["stages"]["diff"]["detail"]
        missing = list(diff["no_script"])
        fetched, unobtainable = [], []
        if missing:
            if not self.online:
                return "fail", "%d card(s) without a script (offline)" % len(missing), {"no_script": missing}
            subprocess.run([sys.executable, FETCHER] + missing, cwd=REPO)
            for name in missing:
                (fetched if resolve_script(name) else unobtainable).append(name)
        new_cards = list(diff["not_in_vocab"]) + fetched
        tokens = sorted(provision_decks.collect_tokens(set(new_cards), CARDS))
        tokens_missing = [t for t in tokens if not os.path.exists(os.path.join(TOKENS, t + ".txt"))]
        tokens_fetched = []
        if tokens_missing and self.online:
            subprocess.run([sys.executable, FETCHER, "--token"] + tokens_missing, cwd=REPO)
            tokens_fetched = [t for t in tokens_missing if os.path.exists(os.path.join(TOKENS, t + ".txt"))]
            tokens_missing = [t for t in tokens_missing if t not in tokens_fetched]
        # A card Forge lacks at the pinned commit is not a failure of this stage: it becomes the
        # capability item script:<card> (stage 5), which blocks stage 6 until FORGE_PIN is bumped
        # (tools/forge_fetch/FORGE_PIN) or the card is otherwise provisioned.
        on_master = [n for n in unobtainable if self.online and forge_master_has(n)]
        detail = {"fetched": fetched, "unobtainable": unobtainable, "on_forge_master": on_master, "tokens": tokens,
                  "tokens_fetched": tokens_fetched, "tokens_missing": tokens_missing}
        summary = "%d fetched, %d not at the Forge pin (%d on master), %d token(s) (%d fetched, %d missing)" % (
            len(fetched), len(unobtainable), len(on_master), len(tokens), len(tokens_fetched), len(tokens_missing))
        return "ok", summary, detail

    # 4 parse
    def stage_parse(self, status):
        diff = status["stages"]["diff"]["detail"]
        prov = status["stages"].get("provision", {}).get("detail", {})
        unprovisioned = sorted(set(diff["no_script"]) - set(prov.get("unobtainable", [])))
        if unprovisioned:
            return "fail", "%d card(s) without a script that provision did not try" % len(unprovisioned), {
                "no_script": unprovisioned}
        cards = {}
        if diff["not_in_vocab"]:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write("\n".join(diff["not_in_vocab"]) + "\n")
                names_file = f.name
            try:
                r = self.engine.run(["--parse-report", "@" + names_file])
            finally:
                os.unlink(names_file)
            if r.returncode not in (0, 1):
                return "fail", "--parse-report exited %d" % r.returncode, {"stderr": r.stderr[-2000:]}
            for line in r.stdout.splitlines():
                rep = json.loads(line)
                if rep.get("card") is None:
                    return "fail", "%s has no script" % rep["query"], {}
                cards[rep["query"]] = {k: rep[k] for k in ("card", "script", "faces", "layout", "vocab", "tokens",
                                                            "findings")}
        n_find = sum(len(c["findings"]) for c in cards.values())
        dirty = sum(1 for c in cards.values() if c["findings"])
        return "ok", "%d new card(s), %d with findings (%d findings), %d without a script" % (
            len(cards), dirty, n_find, len(diff["no_script"])), {"cards": cards, "no_script": sorted(diff["no_script"])}

    # 5 capability
    def stage_capability(self, status):
        parse = status["stages"]["parse"]["detail"]
        cards = dict(parse["cards"])
        for name in parse.get("no_script", []):
            cards[name] = {"findings": [{"kind": "script", "name": name}]}
        needs, mechanics, findings_of = {}, {}, {}
        for name, c in sorted(cards.items()):
            keys = sorted({finding_key(f) for f in c["findings"]})
            mechs = sorted({mechanic_of(self.sizes, k) for k in keys})
            if mechs:
                needs[name] = mechs
            for k in keys:
                findings_of.setdefault(mechanic_of(self.sizes, k), set()).add(k)
            for m in mechs:
                mechanics.setdefault(m, []).append(name)
        clean = sorted(n for n in cards if n not in needs)
        sized = {k: {"cards": v, "size": mechanic_size(self.sizes, k), "findings": sorted(findings_of[k])}
                 for k, v in sorted(mechanics.items())}
        points = sum(size_points(self.sizes, m["size"]) for m in sized.values())
        detail = {"new_cards": sorted(cards), "clean": clean, "needs": needs, "mechanics": sized,
                  "size_points": points}
        return "ok", "%d new card(s): %d clean, %d need %d mechanic(s) (%d points)" % (
            len(cards), len(clean), len(needs), len(sized), points), detail

    # 6 implement
    def stage_implement(self, status):
        cap = status["stages"]["capability"]["detail"]
        if cap["mechanics"]:
            return "pending", "%d mechanic(s) to implement: %s" % (
                len(cap["mechanics"]), ", ".join(sorted(cap["mechanics"]))[:300]), {"mechanics": sorted(cap["mechanics"])}
        return "ok", "every new card parses clean", {}

    # 7 vocab
    def stage_vocab(self, status):
        cards = status["stages"]["parse"]["detail"]["cards"]
        added, tokens = append_vocab(cards, dry_run=self.args.dry_run)
        detail = {"added": added, "tokens": tokens}
        if self.args.dry_run:
            return "pending", "dry run: would add %d entries, %d token(s)" % (len(added), len(tokens)), detail
        if added or tokens:
            r = subprocess.run(["make", "registry"], cwd=REPO)
            detail["make_registry"] = r.returncode
            if r.returncode != 0:
                return "fail", "make registry exited %d" % r.returncode, detail
        print(subprocess.run(["git", "diff", "--stat", "--", "src/card_vocab.h"], cwd=REPO, capture_output=True,
                             text=True).stdout)
        return "ok", "%d entries and %d token(s) appended" % (len(added), len(tokens)), detail

    # 8 text
    def stage_text(self, status):
        r = subprocess.run(["make", "text-table"], cwd=REPO)
        return ("ok" if r.returncode == 0 else "fail"), "make text-table exited %d" % r.returncode, {}

    # 9 checks
    def stage_checks(self, status):
        loc, _ = deck_path(status["deck"])
        config = os.path.basename(os.path.dirname(self.engine.binary))
        detail = {}
        if not self.args.skip_make_check:
            r = subprocess.run(["make", "check"] + (["BUILD=RELEASE"] if config == "release" else []), cwd=REPO)
            detail["make_check"] = r.returncode
            if r.returncode != 0:
                return "fail", "make check exited %d" % r.returncode, detail
        gate = os.path.join(os.path.dirname(self.engine.binary), "robomage_check")
        out_dir = os.path.join(REPO, "ci_out", "onboard_" + status["deck"])
        cmd = [gate, "--tier", "fuzz", "--matchups", "%s/%s:*" % (loc, status["deck"]), "--fuzz-games",
               str(self.args.fuzz_games), "--manual-mana-games", str(self.args.manual_mana_games), "--out-dir", out_dir]
        print("  " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=REPO)
        detail.update({"fuzz": r.returncode, "fuzz_cmd": " ".join(cmd[1:]), "out_dir": os.path.relpath(out_dir, REPO)})
        if r.returncode != 0:
            return "fail", "the fuzz campaign found errors (%s)" % os.path.relpath(out_dir, REPO), detail
        warn_files = []
        for fn in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
            if "_draw_" in fn:
                warn_files.append(fn)
        detail["draws"] = warn_files
        if warn_files:
            return "fail", "%d drawn game(s) to review (%s)" % (len(warn_files), os.path.relpath(out_dir, REPO)), detail
        return "ok", "make check and %d fuzz games per matchup clean" % self.args.fuzz_games, detail

    # 10 designate
    def stage_designate(self, status):
        to = self.args.to
        if to not in ("league", "heldout"):
            return "fail", "designate needs --to league or --to heldout", {}
        loc, path = deck_path(status["deck"])
        if loc == to:
            return "ok", "already in %s/" % to, {"to": to}
        dest = os.path.join(DECKS, to, status["deck"] + ".dk")
        for d in ("league", "heldout"):
            if os.path.exists(os.path.join(DECKS, d, status["deck"] + ".dk")):
                return "fail", "a deck %s/%s.dk exists" % (d, status["deck"]), {}
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = subprocess.run(["git", "mv", path, dest], cwd=REPO)
        if r.returncode != 0:
            return "fail", "git mv exited %d" % r.returncode, {}
        status["location"] = to
        return "ok", "moved to %s/" % to, {"to": to, "from": loc}

    # 11 baseline
    def stage_baseline(self, status):
        loc, _ = deck_path(status["deck"])
        spec = "%s/%s" % (loc, status["deck"])
        pilot = {}
        for fn in sorted(os.listdir(os.path.join(DECKS, "league"))):
            if not fn.endswith(".dk") or fn[:-3] == status["deck"]:
                continue
            opp = "league/" + fn[:-3]
            r = subprocess.run([sys.executable, DRIVER, "--player-a", "scripted", "--player-b", "scripted",
                                "--deck-a", spec, "--deck-b", opp, "--format", "bo1", "--games",
                                str(self.args.baseline_games), "--on-the-play", "random", "--quiet",
                                "--binary", self.engine.binary], cwd=REPO, capture_output=True, text=True)
            m = re.search(r"(\d+)W / (\d+)L / (\d+)D", r.stdout)
            pilot[opp] = {"exit": r.returncode, "w": int(m.group(1)) if m else None, "l": int(m.group(2)) if m else None,
                          "d": int(m.group(3)) if m else None}
        detail = {"scripted_hard": pilot}
        if any(v["exit"] != 0 or v["w"] is None for v in pilot.values()):
            return "fail", "rm_driver.py failed on a matchup", detail
        wins = sum(v["w"] for v in pilot.values())
        games = sum(v["w"] + v["l"] + v["d"] for v in pilot.values())
        summary = "scripted:hard %d/%d with the deck" % (wins, games)
        bench_help = subprocess.run([RMTRAIN, "bench", "--help"], cwd=REPO, capture_output=True, text=True)
        if "--decks" not in bench_help.stdout:
            detail["bench"] = "rmtrain bench has no --decks (docs/open_vocab_plan.md S8 adds it)"
            return "fail", summary + "; rmtrain bench --decks is not available yet (S8)", detail
        if not self.args.run:
            return "fail", summary + "; the benchmark needs --run <run>", detail
        r = subprocess.run([RMTRAIN, "bench", self.args.run, "--decks", spec, "--label", "onboard-" + status["deck"]],
                           cwd=REPO)
        detail["bench_exit"] = r.returncode
        return ("ok" if r.returncode == 0 else "fail"), summary + "; rmtrain bench exited %d" % r.returncode, detail


# ── Mechanics and their sizes ───────────────────────────────────────────────────────────────────

def forge_master_has(name):
    """True when Forge's master branch serves <uid>.txt for `name` (a card newer than the pin)."""
    import urllib.request
    uid = name_to_uid(name)
    url = "https://raw.githubusercontent.com/Card-Forge/forge/master/forge-gui/res/cardsfolder/%s/%s.txt" % (
        uid[0], uid)
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "robomage-onboard/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001  (a 404 or no network: not known to be there)
        return False


def finding_key(finding):
    """A parse finding as "kind:name" ("category:Proliferate")."""
    return "%s:%s" % (finding["kind"], finding["name"])


def mechanic_of(sizes, key):
    """The mechanic a finding key belongs to: the mechanic_sizes.json group listing it ("kind:*"
    lists every name of a kind), else the key itself."""
    for name, m in sizes.get("mechanics", {}).items():
        for k in m.get("keys", []) if isinstance(m, dict) else []:
            if k == key or (k.endswith(":*") and key.startswith(k[:-1])):
                return name
    return key


def load_sizes():
    if not os.path.exists(SIZES_JSON):
        return {"points": {"S": 1, "M": 3, "L": 8, "XL": 20}, "kinds": {}, "mechanics": {}}
    with open(SIZES_JSON, encoding="utf-8") as f:
        return json.load(f)


def mechanic_size(sizes, mechanic):
    m = sizes.get("mechanics", {}).get(mechanic)
    if isinstance(m, dict):
        return m.get("size", "M")
    return sizes.get("kinds", {}).get(mechanic.split(":", 1)[0], "M")


def mechanic_note(sizes, key):
    m = sizes.get("mechanics", {}).get(key)
    return m.get("note", "") if isinstance(m, dict) else ""


def size_points(sizes, size):
    return sizes.get("points", {}).get(size, 3)


# ── Stage 7: the vocab append ───────────────────────────────────────────────────────────────────

def vocab_entries(text):
    cards = [(m.group(1), int(m.group(2))) for m in re.finditer(r'\{"((?:[^"\\]|\\.)*)",\s*(\d+)\}', text)]
    tokens = [(m.group(1), m.group(2), int(m.group(3)))
              for m in re.finditer(r'\{"([a-z0-9_]+)",\s*"((?:[^"\\]|\\.)*)",\s*(\d+)\}', text)]
    return cards, tokens


def token_display_name(stem):
    path = os.path.join(TOKENS, stem + ".txt")
    name = stem
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Name:"):
                    name = line[5:].strip()
                    break
    return name[:-len(" Token")] if name.endswith(" Token") else name


def append_vocab(cards, dry_run=False):
    """Appends each card's faces (a split card: its halves and "A/B", one id) at the next free ids
    below the token band, and each token script it makes at the next free token id, to
    src/card_vocab.h's card_vocab_entries / token_vocab_entries. Returns ({name: id}, {stem: id})."""
    with open(VOCAB_H, encoding="utf-8") as f:
        text = f.read()
    existing, tokens = vocab_entries(text)
    known = {n.lower() for n, _ in existing}
    next_id = max(i for _, i in existing if i < TOKEN_BASE) + 1
    known_tokens = {s for s, _, _ in tokens}
    next_token = max([i for _, _, i in tokens] + [TOKEN_BASE - 1]) + 1
    added, added_tokens, card_lines, token_lines = {}, {}, [], []
    for query in sorted(cards):
        c = cards[query]
        faces = c["faces"]
        # A split card is one card under its halves' names and "A/B" (Dead/Gone); a double-faced
        # card's faces each have an id (Delver of Secrets / Insectile Aberration).
        groups = [list(faces) + ["/".join(faces)]] if c["layout"] == "split" else [[f] for f in faces]
        for group in groups:
            fresh = [n for n in group if n.lower() not in known]
            if not fresh:
                continue
            if next_id >= TOKEN_BASE:
                raise RuntimeError("the card band (ids < %d) is full" % TOKEN_BASE)
            for name in fresh:
                added[name] = next_id
                card_lines.append('    {"%s", %d},' % (name.replace('"', '\\"'), next_id))
                known.add(name.lower())
            next_id += 1
        for t in c["tokens"]:
            stem = t["script"]
            if stem in known_tokens:
                continue
            if next_token > TOKEN_LAST:
                raise RuntimeError("the token band is full")
            added_tokens[stem] = next_token
            token_lines.append('    {"%s", "%s", %d},  // %s' % (stem, token_display_name(stem), next_token, c["card"]))
            known_tokens.add(stem)
            next_token += 1
    if dry_run or (not card_lines and not token_lines):
        return added, added_tokens
    if card_lines:
        end = text.index("};", text.index("card_vocab_entries[]"))
        text = text[:end] + "\n".join(card_lines) + "\n" + text[end:]
    if token_lines:
        end = text.index("};", text.index("token_vocab_entries[]"))
        text = text[:end] + "\n".join(token_lines) + "\n" + text[end:]
    with open(VOCAB_H, "w", encoding="utf-8") as f:
        f.write(text)
    return added, added_tokens


# ── Stage 1: the scrape ─────────────────────────────────────────────────────────────────────────

def scrape(args):
    """Runs the scraper (or imports a scraper output directory with --from) and adds each deck to
    decks/candidates/ with its status file. A stem that names a deck in league/ or heldout/ which
    is not this candidate's gets the suffix "_top8"; a designated candidate is left as it is."""
    work = args.from_dir
    tmp = None
    if not work:
        tmp = tempfile.mkdtemp(prefix="onboard_scrape_")
        work = tmp
        cmd = [sys.executable, SCRAPER, "--top", str(args.top), "--out", work, "--manifest",
               os.path.join(work, "scrape.json"), "--delay", str(args.delay)]
        if args.meta:
            cmd += ["--meta", args.meta]
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            print("scrape: the scraper failed", file=sys.stderr)
            return 1
    with open(os.path.join(work, "scrape.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(STATUS_DIR, exist_ok=True)
    for d in manifest["decks"]:
        stem = d["stem"]
        st = load_status(stem) if os.path.exists(status_path(stem)) else None
        if st is None:
            for other in ("league", "heldout"):
                if os.path.exists(os.path.join(DECKS, other, stem + ".dk")):
                    stem = stem + "_top8"
                    break
            st = load_status(stem)
        if st.get("location") in ("league", "heldout"):
            print("  %-24s scrape     skip designated (%s/)" % (stem, st["location"]))
            continue
        src = os.path.join(work, d["stem"] + ".dk")
        dest = os.path.join(STATUS_DIR, stem + ".dk")
        shutil.copyfile(src, dest)
        st["location"] = "candidates"
        st["source"] = {"site": manifest["site"], "format": manifest["format"], "url": manifest["url"],
                        "window": manifest["window"], "scraped": manifest["date"], "name": d["name"],
                        "rank": d["rank"], "meta_share": d["meta_share"], "section": d["section"],
                        "archetype_id": d["archetype_id"], "deck_id": d["deck_id"], "deck_url": d["deck_url"]}
        record(st, "scrape", "ok", {}, "%s, %s%% (%s), %d main + %d side" % (
            d["name"], d["meta_share"], d["section"], d["main"], d["side"]), {"deck": sha256_file(dest)})
        save_status(st)
        print("  %-24s scrape     ok   %s" % (stem, st["stages"]["scrape"]["summary"]))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


# ── report / REPORT.md ──────────────────────────────────────────────────────────────────────────

def reached(status):
    last = "none"
    for s in STAGES:
        if not stage_ok(status, s):
            break
        last = s
    return last


def next_stage(status):
    for s in STAGES:
        if not stage_ok(status, s):
            rec = status["stages"].get(s)
            return s + ("" if rec is None else " (%s)" % rec["result"])
    return "done"


def deck_row(st):
    src = st.get("source", {})
    stages = st["stages"]
    diff = stages.get("diff", {}).get("detail", {})
    cap = stages.get("capability", {}).get("detail", {})
    new = len(diff.get("not_in_vocab", [])) + len(diff.get("no_script", []))
    return {"deck": st["deck"], "at": st.get("location") or "-", "share": src.get("meta_share"),
            "section": src.get("section", ""), "name": src.get("name", ""), "cards": diff.get("distinct"),
            "new": new if diff else None, "no_script": len(diff.get("no_script", [])) if diff else None,
            "mechanics": len(cap.get("mechanics", {})) if cap else None, "points": cap.get("size_points"),
            "reached": reached(st), "next": next_stage(st)}


def fmt(v):
    return "-" if v is None else str(v)


def report(args):
    rows = [deck_row(load_status(s)) for s in all_stems()]
    rows.sort(key=lambda r: (-(r["share"] or 0), r["deck"]))
    cols = [("deck", "deck"), ("at", "at"), ("share", "share%"), ("section", "section"), ("cards", "cards"),
            ("new", "new"), ("no_script", "no script"), ("mechanics", "mechanics"), ("points", "size"),
            ("reached", "reached"), ("next", "next")]
    widths = [max(len(h), max([len(fmt(r[k])) for r in rows] + [0])) for k, h in cols]
    print("  ".join(h.ljust(w) for (_, h), w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(fmt(r[k]).ljust(w) for (k, _), w in zip(cols, widths)))
    return 0


def write_report_md(sizes):
    statuses = [load_status(s) for s in all_stems()]
    statuses = [s for s in statuses if "capability" in s["stages"]]
    if not statuses:
        return
    rows = sorted((deck_row(s) for s in statuses), key=lambda r: (r["points"] or 0, r["new"] or 0, r["deck"]))
    by_deck = {s["deck"]: s for s in statuses}
    mech_decks, mech_cards, mech_findings = {}, {}, {}
    for s in statuses:
        for k, m in s["stages"]["capability"]["detail"]["mechanics"].items():
            mech_decks.setdefault(k, set()).add(s["deck"])
            mech_cards.setdefault(k, set()).update(m["cards"])
            mech_findings.setdefault(k, set()).update(m.get("findings", []))
    sole = {k: sorted(d for d in decks if len(by_deck[d]["stages"]["capability"]["detail"]["mechanics"]) == 1)
            for k, decks in mech_decks.items()}
    src = [s.get("source", {}) for s in statuses if s.get("source")]
    dates = sorted({x.get("scraped", "") for x in src})
    windows = sorted({x.get("window", "") for x in src})
    out = ["# Candidate decks: capability report", "",
           "Generated by `python3 tools/onboard/onboard.py capability` from the status files beside it "
           "(do not edit by hand; `tools/onboard/README.md`). Scraped %s from MTGTop8 Legacy (%s), %d decks."
           % (", ".join(dates) or "-", ", ".join(windows) or "-", len(statuses)), "",
           "Size estimates (`tools/onboard/mechanic_sizes.json`): S ≤ ½ day (a param or a flag), M ≈ 1 day (an "
           "effect handler, a trigger mode, a keyword), L 2–3 days (a rules subsystem), XL a week or more; "
           "points S %d, M %d, L %d, XL %d." % tuple(size_points(sizes, x) for x in ("S", "M", "L", "XL")), "",
           "## Decks (cheapest first)", "",
           "| deck | MTGTop8 name | share % | section | cards | new cards | mechanics | size points | reached |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["deck"], r["name"], fmt(r["share"]), r["section"], fmt(r["cards"]), fmt(r["new"]), fmt(r["mechanics"]),
            fmt(r["points"]), r["reached"]))
    out += ["", "## Mechanics ranked by the decks that need them", "",
            "A deck is unlocked when every mechanic it needs exists; *sole blocker of* lists the decks this one "
            "mechanic alone unlocks.", "",
            "| mechanic | size | decks needing it | sole blocker of | cards | findings | note |",
            "|---|---|---|---|---|---|---|"]
    order = sorted(mech_decks, key=lambda k: (-len(mech_decks[k]), size_points(sizes, mechanic_size(sizes, k)), k))
    for k in order:
        cards = sorted(mech_cards[k])
        found = sorted(mech_findings[k] - {k})
        out.append("| %s | %s | %d | %s | %s | %s | %s |" % (
            k, mechanic_size(sizes, k), len(mech_decks[k]), ", ".join(sole[k]) or "-",
            ", ".join(cards[:6]) + (" (+%d)" % (len(cards) - 6) if len(cards) > 6 else ""),
            ", ".join("`%s`" % f for f in found) or "-", mechanic_note(sizes, k)))
    out += ["", "## Unlock order (greedy: the next cheapest deck given what is already implemented)", "",
            "| step | deck | new mechanics | their points | cumulative points | decks playable |", "|---|---|---|---|---|---|"]
    done, total, playable, step = set(), 0, 0, 0
    remaining = {s["deck"]: set(s["stages"]["capability"]["detail"]["mechanics"]) for s in statuses}
    while remaining:
        def cost(d):
            return sum(size_points(sizes, mechanic_size(sizes, k)) for k in remaining[d] - done)
        d = min(remaining, key=lambda x: (cost(x), x))
        new = sorted(remaining[d] - done)
        c = cost(d)
        done |= remaining.pop(d)
        total += c
        playable += 1
        step += 1
        out.append("| %d | %s | %s | %d | %d | %d |" % (step, d, "; ".join(new) or "-", c, total, playable))
    out += ["", "## Per deck", ""]
    for r in rows:
        s = by_deck[r["deck"]]
        cap = s["stages"]["capability"]["detail"]
        out.append("### %s — %s (%s%%, %s)" % (r["deck"], r["name"], fmt(r["share"]), r["section"]))
        out.append("")
        out.append("%s cards, %d new; clean: %s" % (fmt(r["cards"]), len(cap["new_cards"]),
                                                    ", ".join(cap["clean"]) or "-"))
        out.append("")
        for card, keys in sorted(cap["needs"].items()):
            out.append("- %s: %s" % (card, "; ".join(keys)))
        out.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out).rstrip() + "\n")
    print("wrote %s" % os.path.relpath(REPORT_MD, REPO))


# ── check (the gate's offline part) ─────────────────────────────────────────────────────────────

def check(args):
    """Reruns diff, parse and capability for every candidate in memory (no file written, no
    network) and compares the results with the committed status files: a tool failure is an
    error (exit 1), a status file whose recorded result differs is reported as stale."""
    pipe = Pipeline(args)
    pipe.write = False
    pipe.online = False
    errors, stale, n = 0, [], 0
    for stem in all_stems():
        if not os.path.exists(status_path(stem)):
            continue
        committed = load_status(stem)
        if committed.get("location") != "candidates":
            continue
        n += 1
        st = json.loads(json.dumps(committed))
        st["stages"] = {k: v for k, v in st["stages"].items() if k in ("scrape", "provision")}
        for stage in ("diff", "parse", "capability"):
            inputs = pipe.inputs(stage, st)
            result, summary, detail = getattr(pipe, "stage_" + stage)(st)
            record(st, stage, result, inputs, summary, detail)
            if result != "ok":
                print("onboard check: ERROR %s: %s %s: %s" % (stem, stage, result, summary))
                errors += 1
                break
            old = committed["stages"].get(stage, {})
            if old.get("result") != result or old.get("detail") != detail:
                stale.append("%s (%s)" % (stem, stage))
                break
    for s in stale:
        print("onboard check: STALE %s: the committed status file differs; rerun onboard.py all" % s)
    print("onboard check: %d candidate(s), %d error(s), %d stale" % (n, errors, len(stale)))
    return 1 if errors else 0


# ── main ────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=STAGES + ["all", "report", "check"])
    ap.add_argument("--deck", action="append", default=[], help="a deck stem (repeatable; default every deck "
                                                                "with a status file or in decks/candidates/)")
    ap.add_argument("--force", action="store_true", help="rerun the named stage even when it is current")
    ap.add_argument("--binary", help="the engine (default bin/debug/robomage; ROBOMAGE_BUILD=release for "
                                     "bin/release/robomage)")
    ap.add_argument("--offline", action="store_true", help="provision fetches nothing (a missing script fails)")
    ap.add_argument("--top", type=int, default=25, help="scrape: the number of decks (default 25)")
    ap.add_argument("--meta", default="39", help="scrape: MTGTop8's time window id (default 39, the last 2 months)")
    ap.add_argument("--delay", type=float, default=1.0, help="scrape: seconds between requests (default 1.0)")
    ap.add_argument("--from", dest="from_dir", help="scrape: import a scrape_meta.py --manifest output directory "
                                                    "instead of fetching")
    ap.add_argument("--dry-run", action="store_true", help="vocab: print the entries without editing")
    ap.add_argument("--to", help="designate: league or heldout")
    ap.add_argument("--skip-make-check", action="store_true", help="checks: only the fuzz campaign")
    ap.add_argument("--fuzz-games", type=int, default=100, help="checks: bo1 games per matchup (default 100)")
    ap.add_argument("--manual-mana-games", type=int, default=10,
                    help="checks: --no-auto-mana games per matchup (default 10)")
    ap.add_argument("--baseline-games", type=int, default=10,
                    help="baseline: scripted:hard games per league opponent (default 10)")
    ap.add_argument("--run", help="baseline: the run whose policies rmtrain bench plays")
    args = ap.parse_args(argv)

    if args.command == "scrape":
        return scrape(args)
    if args.command == "report":
        return report(args)
    if args.command == "check":
        return check(args)
    pipe = Pipeline(args)
    pipe.write = not args.dry_run
    stems = args.deck or all_stems()
    stages = AUTOMATIC if args.command == "all" else [args.command]
    failed = 0
    for stem in stems:
        if deck_path(stem)[0] is None:
            print("  %s: no such deck in %s" % (stem, ", ".join(DECK_DIRS)))
            failed += 1
            continue
        status = load_status(stem)
        for stage in stages:
            result = pipe.ensure(stage, status, force=args.force and args.command != "all")
            if result != "ok":
                failed += result not in ("pending",)
                break
    if args.command in ("all", "capability", "implement") or "capability" in stages:
        write_report_md(pipe.sizes)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
