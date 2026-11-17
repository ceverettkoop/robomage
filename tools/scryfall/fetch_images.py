#!/usr/bin/env python3
"""Fetch card images from Scryfall into the workbench's image cache.

Standalone tool, Python standard library only. For every vocab card (src/card_vocab.h, one
entry per vocab index) and every token script the decks can make (bin/resources/tokenscripts/,
which tools/forge_fetch/provision_decks.py provisions), it lists the card's paper printings from
the Scryfall API, chooses one (the rule below), downloads that printing's `normal` image (each
face of a double-faced card) into the cache and writes the cache's manifest.json. The workbench
(tools/rmwork, docs/workbench_plan.md §2.4) reads only the cache; it never touches the network.

    python3 tools/scryfall/fetch_images.py                      # the whole vocab and the tokens
    python3 tools/scryfall/fetch_images.py --cards "Lightning Bolt" "Delver of Secrets"
    python3 tools/scryfall/fetch_images.py --printing "Lightning Bolt=m10:146"
    python3 tools/scryfall/fetch_images.py --dry-run            # list, choose, print; write nothing
    python3 tools/scryfall/fetch_images.py --selftest           # the choice rule on canned JSON

The cache (default $XDG_CACHE_HOME/robomage/scryfall, else ~/.cache/robomage/scryfall; --cache):

    manifest.json                 what the workbench reads (below)
    images/<set>/<number>.jpg     a printing's image; a double-faced card's faces <number>-0.jpg, -1.jpg

The default printing: the OLDEST printing (released date, then set code, then collector number)
that is printed on paper ("paper" among its games), not digital, not a promo, not oversized,
not from a set of type memorabilia, funny, minigame, alchemy, vanguard, planechase, archenemy or
treasure_chest (gold-bordered World Championship decks, Collectors' Edition, un-sets, oversized
cards), and whose image Scryfall has (image_status not missing / placeholder). When no printing
passes, the oldest paper printing with an image is chosen. A token is searched by its script's
name, colours, power / toughness, artifact type and keywords (relaxed in that order — keywords,
then P/T, then colours — until a token matches) and chosen by the same rule. The printing list
kept for the Settings picker is every paper printing (promos and the excluded set types too).

manifest.json:
    {"version": 1, "rule": "...", "updated": "<UTC time>",
     "cards": [{"key": "Delver of Secrets",      # the vocab name (a token: "token:<script stem>")
                "index": 16,                     # vocab index (an unregistered token: -1)
                "token": false, "stem": "",      # a token's script stem
                "names": ["Delver of Secrets"],  # every vocab name of the index
                "scryfall_name": "Delver of Secrets // Insectile Aberration",
                "chosen": 0,                     # index into printings
                "printings": [{"set": "isd", "set_name": "Innistrad", "number": "51",
                               "released": "2011-09-30", "set_type": "expansion",
                               "promo": false, "default_ok": true,
                               "faces": [{"name": "Delver of Secrets", "url": "https://…",
                                          "file": "images/isd/51-0.jpg"}, …]}, …]}, …],
     "missing": [{"key": "...", "why": "..."}]}

A face's "file" is where its image goes in the cache; the workbench checks the file exists.

Scryfall's API rules are kept: a generic User-Agent and Accept header, API requests 500 ms apart
(its search and named-card endpoints take two a second) and image downloads 100 ms apart, a retry
with backoff on 429 and 5xx (Retry-After honoured). A run is resumable: a card
already in the manifest is not listed again (unless named with --cards or --refresh), an image
already in the cache is not downloaded again. Exit code: the number of cards or tokens with no
image (0 = all), 2 on a usage error.
"""

import argparse
import datetime
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.scryfall.com"
USER_AGENT = "robomage-workbench/1.0"
# Scryfall asks for 50-100 ms between requests and limits /cards/search and /cards/named to two a
# second (a 429 then costs a minute's Retry-After): API requests are spaced 500 ms apart, image
# downloads from its CDN 100 ms.
API_INTERVAL_S = 0.5
IMAGE_INTERVAL_S = 0.1
RETRIES = 6
TIMEOUT_S = 30

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOCAB_H = os.path.join(_REPO_ROOT, "src", "card_vocab.h")
TOKENS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "tokenscripts")

EXCLUDED_SET_TYPES = {"memorabilia", "funny", "minigame", "alchemy", "vanguard", "planechase", "archenemy",
                      "treasure_chest"}
RULE = ("oldest paper printing (released date, set code, collector number) that is not digital, not a promo, "
        "not oversized, not from a memorabilia / funny / minigame / alchemy / vanguard / planechase / archenemy / "
        "treasure_chest set, with an image; else the oldest paper printing with an image")
COLOR_LETTERS = {"white": "w", "blue": "u", "black": "b", "red": "r", "green": "g"}


# ── Choosing a printing (pure: --selftest checks it on canned JSON) ─────────────────────────────

def has_image(card):
    """Whether Scryfall has an image of the printing."""
    if card.get("image_status") in ("missing", "placeholder"):
        return False
    return bool(card.get("image_uris")) or any(f.get("image_uris") for f in card.get("card_faces", []))


def is_paper(card):
    return "paper" in card.get("games", []) and not card.get("digital", False)


def default_ok(card, token=False):
    """Whether the printing may be the default one (the rule in the module docstring)."""
    if not is_paper(card) or not has_image(card):
        return False
    if card.get("promo", False) or card.get("oversized", False):
        return False
    st = card.get("set_type", "")
    if st in EXCLUDED_SET_TYPES:
        return False
    if not token and st == "token":
        return False
    return True


def number_key(number):
    """Collector numbers in natural order: "2" < "10" < "10a" < "★1"."""
    m = re.match(r"(\d+)(.*)", number or "")
    return (0, int(m.group(1)), m.group(2)) if m else (1, 0, number or "")


def printing_order(card):
    return (card.get("released_at", "9999-99-99"), card.get("set", ""), number_key(card.get("collector_number", "")))


def choose_printing(cards, token=False):
    """The default printing among `cards` (Scryfall card objects), or None."""
    paper = sorted((c for c in cards if is_paper(c) and has_image(c)), key=printing_order)
    for c in paper:
        if default_ok(c, token):
            return c
    return paper[0] if paper else None


def safe_part(text):
    """A collector number or set code as a file name part."""
    out = re.sub(r"[^A-Za-z0-9._-]", lambda m: "_%02x" % ord(m.group(0)[0]) if ord(m.group(0)[0]) < 256 else "_u",
                 text or "")
    return out or "_"


def fold(name):
    """A name compared accent- and case-insensitively."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()


def printing_faces(card, face_name=None):
    """The printing's image faces: [{"name", "url", "file"}]. A card with one image (single-faced,
    split, adventure, flip) has one; a double-faced card one per face. `face_name` keeps only
    that face of a double-faced token."""
    base = "images/%s/%s" % (safe_part(card.get("set", "")), safe_part(card.get("collector_number", "")))
    if card.get("image_uris"):
        return [{"name": card.get("name", ""), "url": card["image_uris"].get("normal", ""), "file": base + ".jpg"}]
    faces = []
    for i, f in enumerate(card.get("card_faces", [])):
        if not f.get("image_uris"):
            continue
        faces.append({"name": f.get("name", ""), "url": f["image_uris"].get("normal", ""),
                      "file": "%s-%d.jpg" % (base, i)})
    if face_name is not None:
        same = [f for f in faces if fold(f["name"]) == fold(face_name)]
        if same:
            return same
    return faces


def printing_entry(card, token=False, face_name=None):
    return {
        "set": card.get("set", ""),
        "set_name": card.get("set_name", ""),
        "number": card.get("collector_number", ""),
        "released": card.get("released_at", ""),
        "set_type": card.get("set_type", ""),
        "promo": bool(card.get("promo", False)),
        "default_ok": default_ok(card, token),
        "faces": printing_faces(card, face_name),
    }


def build_entry(key, index, names, cards, token=False, stem="", face_name=None):
    """The manifest entry of a card or token from its Scryfall printings (None when it has none)."""
    paper = sorted((c for c in cards if is_paper(c)), key=printing_order)
    chosen = choose_printing(paper, token)
    if chosen is None:
        return None
    printings = [printing_entry(c, token, face_name) for c in paper]
    return {
        "key": key,
        "index": index,
        "token": token,
        "stem": stem,
        "names": names,
        "scryfall_name": chosen.get("name", ""),
        "chosen": paper.index(chosen),
        "printings": printings,
    }


# ── The vocab and the token scripts ──────────────────────────────────────────────────────────────

def read_vocab(path=VOCAB_H):
    """[(index, [names])] of card_vocab_entries, by first appearance; and {stem: index} of the
    token band."""
    text = open(path, encoding="utf-8").read()
    cards_part = text[text.index("card_vocab_entries[] = {"):text.index("CARD_VOCAB_SIZE =")]
    by_index = {}
    order = []
    for name, idx in re.findall(r'\{"((?:[^"\\]|\\.)*)",\s*(\d+)\}', cards_part):
        idx = int(idx)
        if idx not in by_index:
            by_index[idx] = []
            order.append(idx)
        by_index[idx].append(name.replace('\\"', '"'))
    tokens_part = text[text.index("token_vocab_entries[] = {"):text.index("TOKEN_VOCAB_SIZE =")]
    token_index = {stem: int(idx) for stem, _, idx in
                   re.findall(r'\{"([a-z0-9_]+)",\s*"([^"]*)",\s*(\d+)\}', tokens_part)}
    return [(i, by_index[i]) for i in order], token_index


def vocab_key(names):
    """The vocab name a card is known by: a split card's "Front/Back" name, else the first."""
    for n in names:
        if "/" in n:
            return n
    return names[0]


def read_token_script(path):
    """A token script's name (without " Token"), colours, types, P/T and keywords."""
    info = {"name": "", "colors": "", "types": [], "power": None, "toughness": None, "keywords": []}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("Name:"):
            info["name"] = re.sub(r"\s+Token$", "", line[5:].strip())
        elif line.startswith("Colors:"):
            info["colors"] = "".join(COLOR_LETTERS.get(c.strip(), "") for c in line[7:].split(","))
        elif line.startswith("Types:"):
            info["types"] = line[6:].split()
        elif line.startswith("PT:"):
            p, _, t = line[3:].partition("/")
            info["power"], info["toughness"] = p.strip(), t.strip()
        elif line.startswith("K:"):
            info["keywords"].append(line[2:].split(":")[0].strip())
    return info


def token_queries(info):
    """Scryfall searches for a token, strictest first: name, colours, types, P/T, keywords;
    then without keywords, without P/T, without colours."""
    name = 't:token !"%s"' % info["name"]
    colors = "c=%s" % (info["colors"] or "c")
    types = " ".join("t:%s" % t.lower() for t in info["types"] if t in ("Artifact", "Creature", "Enchantment",
                                                                         "Planeswalker", "Legendary"))
    pt = ""
    if info["power"] is not None and re.fullmatch(r"\d+", info["power"] or "") and \
            re.fullmatch(r"\d+", info["toughness"] or ""):
        pt = "pow=%s tou=%s" % (info["power"], info["toughness"])
    kws = " ".join('keyword:"%s"' % k.lower() for k in info["keywords"] if re.fullmatch(r"[A-Za-z ]+", k))
    out = []
    for q in (" ".join(p for p in (name, colors, types, pt, kws) if p),
              " ".join(p for p in (name, colors, types, pt) if p),
              " ".join(p for p in (name, colors, types) if p),
              " ".join(p for p in (name, types) if p)):
        if q not in out:
            out.append(q)
    return out


# ── HTTP ─────────────────────────────────────────────────────────────────────────────────────────

class Client:
    """Scryfall requests spaced API_INTERVAL_S apart (images IMAGE_INTERVAL_S), retried with backoff
    on 429 / 5xx."""

    def __init__(self, log):
        self.last = {}
        self.requests = 0
        self.log = log

    def _wait(self, url):
        api = url.startswith(API)
        interval = API_INTERVAL_S if api else IMAGE_INTERVAL_S
        delay = self.last.get(api, 0.0) + interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last[api] = time.monotonic()

    def get(self, url, accept="application/json"):
        """The body of a 200 response; None for a 404; raises RuntimeError after the retries."""
        backoff = 1.0
        for attempt in range(RETRIES):
            self._wait(url)
            self.requests += 1
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code != 429 and e.code < 500:
                    raise RuntimeError("%s: HTTP %d" % (url, e.code))
                wait = backoff
                try:
                    wait = max(wait, float(e.headers.get("Retry-After", "0")))
                except ValueError:
                    pass
                self.log("  HTTP %d, retrying in %.0f s" % (e.code, wait))
            except (urllib.error.URLError, OSError) as e:
                wait = backoff
                self.log("  %s, retrying in %.0f s" % (getattr(e, "reason", e), wait))
            time.sleep(wait)
            backoff = min(backoff * 2, 60.0)
        raise RuntimeError("%s: gave up after %d attempts" % (url, RETRIES))

    def json(self, path, params=None):
        url = path if path.startswith("http") else API + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        body = self.get(url)
        return None if body is None else json.loads(body.decode("utf-8"))

    def search(self, query):
        """Every printing a search finds, oldest first (each page followed)."""
        page = self.json("/cards/search", {"q": query, "unique": "prints", "order": "released", "dir": "asc"})
        cards = []
        while page:
            cards.extend(page.get("data", []))
            if not page.get("has_more"):
                break
            page = self.json(page["next_page"])
        return cards


def lookup_card(client, names):
    """(Scryfall card, the name found) for the first of `names` Scryfall knows, by exact name then
    fuzzy; (None, why) when none."""
    tries = []
    for n in names:
        tries.append(n.replace("/", " // ") if "/" in n and "//" not in n else n)
    for kind in ("exact", "fuzzy"):
        for n in tries:
            card = client.json("/cards/named", {kind: n})
            if card and card.get("object") == "card":
                return card, n
    return None, "Scryfall knows no card named " + " / ".join(names)


def card_printings(client, card):
    """Every paper printing of `card`'s oracle card."""
    oid = card.get("oracle_id") or (card.get("card_faces") or [{}])[0].get("oracle_id")
    if not oid:
        return [card]
    return client.search("oracleid:%s game:paper" % oid)


# ── The cache ────────────────────────────────────────────────────────────────────────────────────

def default_cache():
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "robomage", "scryfall")


def load_manifest(cache):
    path = os.path.join(cache, "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if isinstance(m, dict) and isinstance(m.get("cards"), list):
            return m
    except (OSError, ValueError):
        pass
    return {"version": 1, "cards": [], "missing": []}


def save_manifest(cache, manifest):
    os.makedirs(cache, exist_ok=True)
    manifest["version"] = 1
    manifest["rule"] = RULE
    manifest["updated"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", dir=cache)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, os.path.join(cache, "manifest.json"))


def download_faces(client, cache, printing, dry_run, log):
    """Downloads the printing's face images not yet cached; returns how many faces are cached."""
    have = 0
    for face in printing["faces"]:
        dest = os.path.join(cache, face["file"])
        if os.path.exists(dest):
            have += 1
            continue
        if dry_run or not face.get("url"):
            continue
        body = client.get(face["url"], accept="image/*")
        if body is None:
            log("  %s: image not found (%s)" % (face["name"], face["url"]))
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".img.", dir=os.path.dirname(dest))
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        os.replace(tmp, dest)
        have += 1
    return have


# ── The run ──────────────────────────────────────────────────────────────────────────────────────

def targets(vocab, token_index, tokens_dir, only):
    """[(key, index, names, token, stem, script info)] to fetch; `only` (folded names) limits it."""
    out = []
    for idx, names in vocab:
        key = vocab_key(names)
        if only and not any(fold(n) in only for n in names + [key]):
            continue
        out.append((key, idx, names, False, "", None))
    if os.path.isdir(tokens_dir):
        for fn in sorted(os.listdir(tokens_dir)):
            if not fn.endswith(".txt"):
                continue
            stem = fn[:-4]
            info = read_token_script(os.path.join(tokens_dir, fn))
            key = "token:" + stem
            if only and not ({fold(key), fold(stem), fold(info["name"])} & only):
                continue
            out.append((key, token_index.get(stem, -1), [info["name"]], True, stem, info))
    return out


def list_entry(client, key, idx, names, token, stem, info, log):
    """The manifest entry for a target (None with the reason)."""
    if token:
        for q in token_queries(info):
            cards = client.search(q)
            if cards:
                entry = build_entry(key, idx, names, cards, True, stem, info["name"])
                if entry:
                    return entry, ""
        return None, "Scryfall has no paper token %s" % info["name"]
    card, found = lookup_card(client, names)
    if card is None:
        return None, found
    entry = build_entry(key, idx, names, card_printings(client, card))
    if entry is None:
        return None, "no paper printing with an image"
    return entry, ""


def run(args):
    log = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))
    cache = os.path.abspath(os.path.expanduser(args.cache or default_cache()))
    vocab, token_index = read_vocab(args.vocab)
    only = {fold(n) for n in (args.cards or [])}
    overrides = []
    for p in args.printing or []:
        name, sep, rest = p.partition("=")
        set_code, sep2, number = rest.partition(":")
        if not sep or not sep2 or not name or not set_code or not number:
            print("fetch_images: --printing wants NAME=SET:NUMBER, not %r" % p, file=sys.stderr)
            return 2
        overrides.append((name, set_code.lower(), number))
        only.add(fold(name))
    client = Client(log)
    manifest = load_manifest(cache)
    by_key = {e["key"]: e for e in manifest["cards"]}
    missing = {m["key"]: m for m in manifest.get("missing", [])}
    todo = targets(vocab, token_index, args.tokens_dir, only)
    if only and not todo:
        print("fetch_images: no vocab card or token named %s" % ", ".join(sorted(only)), file=sys.stderr)
        return 2
    failed = 0
    images = 0
    for n, (key, idx, names, token, stem, info) in enumerate(todo, 1):
        entry = by_key.get(key)
        relist = entry is None or args.refresh or (args.cards and not overrides)
        try:
            if relist:
                log("[%d/%d] %s" % (n, len(todo), key))
                entry, why = list_entry(client, key, idx, names, token, stem, info, log)
                if entry is None:
                    log("  MISSING: " + why)
                    missing[key] = {"key": key, "why": why}
                    by_key.pop(key, None)
                    failed += 1
                    continue
                missing.pop(key, None)
                by_key[key] = entry
            entry["index"] = idx
            entry["names"] = names
            wanted = [entry["printings"][entry["chosen"]]]
            for name, set_code, number in overrides:
                if fold(name) in {fold(x) for x in names + [key, stem] if x}:
                    match = [p for p in entry["printings"] if p["set"] == set_code and p["number"] == number]
                    if not match:
                        card = client.json("/cards/%s/%s" % (set_code, urllib.parse.quote(number)))
                        if not card or card.get("object") != "card":
                            print("fetch_images: Scryfall has no printing %s:%s" % (set_code, number), file=sys.stderr)
                            failed += 1
                            continue
                        match = [printing_entry(card, token, info["name"] if token else None)]
                        entry["printings"].append(match[0])
                        entry["printings"].sort(key=lambda p: (p["released"], p["set"], number_key(p["number"])))
                        entry["chosen"] = entry["printings"].index(wanted[0])
                    wanted = match
                    log("  printing %s:%s" % (set_code, number))
            for p in wanted:
                have = download_faces(client, cache, p, args.dry_run, log)
                images += have
                if have == 0 and not args.dry_run:
                    failed += 1
            if args.dry_run and relist:
                c = entry["printings"][entry["chosen"]]
                log("  %s: %s (%s) #%s, %s; %d paper printings" % (entry["scryfall_name"], c["set_name"], c["set"],
                                                                   c["number"], c["released"], len(entry["printings"])))
        except RuntimeError as e:
            log("  ERROR: %s" % e)
            failed += 1
        if not args.dry_run and n % 25 == 0:
            manifest["cards"] = list(by_key.values())
            manifest["missing"] = list(missing.values())
            save_manifest(cache, manifest)
    manifest["cards"] = sorted(by_key.values(), key=lambda e: (e["token"], e["index"] if e["index"] >= 0 else 1 << 20,
                                                               e["key"]))
    manifest["missing"] = sorted(missing.values(), key=lambda m: m["key"])
    if not args.dry_run:
        save_manifest(cache, manifest)
    log("fetch_images: %d entries, %d images cached or fetched this run, %d without an image, %d requests; cache %s"
        % (len(by_key), images, failed, client.requests, cache))
    for m in manifest["missing"]:
        log("  missing: %s (%s)" % (m["key"], m["why"]))
    return min(failed, 125)


# ── --selftest: the choice rule on canned Scryfall JSON ─────────────────────────────────────────

def _canned(set_code, number, released, **kw):
    card = {"object": "card", "name": kw.pop("name", "Lightning Bolt"), "set": set_code, "set_name": set_code.upper(),
            "collector_number": number, "released_at": released, "games": kw.pop("games", ["paper"]),
            "digital": False, "promo": False, "oversized": False, "set_type": "core", "image_status": "highres_scan",
            "image_uris": {"normal": "https://img.example/%s/%s.jpg" % (set_code, number)}}
    card.update(kw)
    return card


def selftest():
    failures = []

    def check(ok, what):
        if not ok:
            failures.append(what)

    prints = [
        _canned("ced", "161", "1993-12-01", set_type="memorabilia"),
        _canned("lea", "161", "1993-08-05"),
        _canned("pmei", "1", "1990-01-01", promo=True),
        _canned("leb", "162", "1993-10-04"),
        _canned("mtgo", "5", "1980-01-01", games=["mtgo"], digital=True),
        _canned("unh", "1", "1985-01-01", set_type="funny"),
        _canned("o90", "1", "1989-01-01", oversized=True),
        _canned("wc97", "1", "1988-01-01", set_type="memorabilia"),
    ]
    c = choose_printing(prints)
    check(c is not None and c["set"] == "lea", "the oldest non-promo paper printing is Alpha, not %s" % (c or {}).get("set"))
    e = build_entry("Lightning Bolt", 2, ["Lightning Bolt"], prints)
    check(e is not None and e["printings"][e["chosen"]]["set"] == "lea", "the entry's chosen printing is not Alpha")
    check(e is not None and all(p["set"] != "mtgo" for p in e["printings"]), "a digital printing is listed")
    check(e is not None and len(e["printings"]) == 7, "not every paper printing is listed")
    check(e is not None and [p["set"] for p in e["printings"]] == ["unh", "wc97", "o90", "pmei", "lea", "leb", "ced"],
          "printings not oldest first")
    check(e is not None and e["printings"][e["chosen"]]["faces"][0]["file"] == "images/lea/161.jpg",
          "a single-faced printing's file")
    # Same date: set code, then collector number in natural order.
    tie = [_canned("m10", "10", "2009-07-17"), _canned("m10", "9", "2009-07-17"), _canned("m10", "9a", "2009-07-17")]
    check(choose_printing(tie)["collector_number"] == "9", "collector numbers are not in natural order")
    # Only excluded printings: the oldest paper one with an image.
    only_promo = [_canned("pmei", "2", "2001-01-01", promo=True), _canned("plst", "1", "2000-01-01", promo=True)]
    check(choose_printing(only_promo)["set"] == "plst", "no fallback to the oldest paper printing")
    # No image: never chosen.
    no_img = [_canned("lea", "1", "1993-08-05", image_status="missing"), _canned("leb", "1", "1993-10-04")]
    check(choose_printing(no_img)["set"] == "leb", "a printing without an image was chosen")
    check(choose_printing([_canned("mtgo", "1", "2000-01-01", games=["mtgo"], digital=True)]) is None,
          "a digital-only card has a printing")
    # A double-faced card: a face each.
    dfc = _canned("isd", "51", "2011-09-30", name="Delver of Secrets // Insectile Aberration")
    del dfc["image_uris"]
    dfc["card_faces"] = [{"name": "Delver of Secrets", "image_uris": {"normal": "https://img.example/f.jpg"}},
                         {"name": "Insectile Aberration", "image_uris": {"normal": "https://img.example/b.jpg"}}]
    faces = printing_faces(dfc)
    check([f["file"] for f in faces] == ["images/isd/51-0.jpg", "images/isd/51-1.jpg"], "a DFC's face files")
    check([f["name"] for f in faces] == ["Delver of Secrets", "Insectile Aberration"], "a DFC's face names")
    check(printing_faces(dfc, "Insectile Aberration")[0]["file"] == "images/isd/51-1.jpg",
          "a double-faced token's own face")
    # A split card: one image.
    split = _canned("akh", "1", "2017-04-28", name="Dead // Gone")
    split["card_faces"] = [{"name": "Dead"}, {"name": "Gone"}]
    check(len(printing_faces(split)) == 1, "a split card has one image")
    # Tokens: token sets allowed, the queries relax.
    tok = _canned("tsoi", "1", "2016-04-08", name="Clue", set_type="token")
    check(choose_printing([tok], token=True)["set"] == "tsoi", "a token printing is refused")
    check(not default_ok(tok, token=False), "a token set passes for a card")
    q = token_queries({"name": "Construct", "colors": "", "types": ["Artifact", "Creature", "Construct"],
                       "power": "4", "toughness": "4", "keywords": ["Flying", "Haste"]})
    check(q[0] == 't:token !"Construct" c=c t:artifact t:creature pow=4 tou=4 keyword:"flying" keyword:"haste"',
          "the strictest token query: %s" % q[0])
    check(q[-1] == 't:token !"Construct" t:artifact t:creature', "the loosest token query: %s" % q[-1])
    check(len(token_queries({"name": "Illusion", "colors": "u", "types": ["Creature", "Illusion"], "power": "*",
                             "toughness": "*", "keywords": []})) == 2, "a */* token's queries")
    check(safe_part("★12") == "_u12" and safe_part("12a") == "12a", "file name parts")
    # The vocab: split cards keep one key, the token band reads.
    vocab, token_index = read_vocab()
    by = dict(vocab)
    check(vocab_key(by.get(328, ["?"])) == "Dead/Gone", "Dead/Gone's key")
    check(token_index.get("c_a_clue_draw") == 906, "the token band")
    check(fold("Lórien Revealed") == "lorien revealed", "accent folding")
    for f in failures:
        print("FAIL: " + f)
    print("FETCH-IMAGES-SELFTEST: %d failures" % len(failures))
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", help="the image cache (default %s)" % default_cache())
    ap.add_argument("--cards", nargs="+", metavar="NAME", help="only these cards or tokens (listed again)")
    ap.add_argument("--printing", action="append", metavar="NAME=SET:NUMBER",
                    help="fetch this printing's images (repeatable)")
    ap.add_argument("--refresh", action="store_true", help="list every card's printings again")
    ap.add_argument("--dry-run", action="store_true", help="list and choose, print, write nothing")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="check the choice rule on canned JSON (no network)")
    ap.add_argument("--vocab", default=VOCAB_H, help=argparse.SUPPRESS)
    ap.add_argument("--tokens-dir", default=TOKENS_DIR, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
