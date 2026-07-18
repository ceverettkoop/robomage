"""Scryfall card-image download + on-disk cache for the PySide6 game board.

Qt-free by design (``requests`` + ``threading`` + stdlib only) so it is
independently testable and importable from a headless context. The GUI
(``train/gui_game.py``) wraps this in an ``ImageProvider`` QObject; nothing in
here touches Qt.

Images are the real Scryfall "normal" art (488x680 — sharp at both the ~112 px
board card size and a larger oracle-popup), cached under
``bin/resources/card_images/`` as ``<uid>.jpg`` (uid via ``decode._name_to_uid``,
mirroring the engine's card filenames). Tokens are keyed with their power and
toughness (``token_<uid>_<p>_<t>``) so different-sized same-name tokens don't
collide.

Politeness: a single daemon worker thread drains a FIFO queue (with a dedup set)
and a monotonic-clock throttle enforces >=120 ms between HTTP requests (Scryfall
asks for <10 req/s). Definitive lookup failures (a real 404 / no search match)
are recorded in ``misses.json`` so a nonsense name isn't re-queried every launch;
transient network errors are NOT recorded, so an offline run never poisons the
miss manifest. (Delete ``misses.json`` to retry recorded misses.)

Public API:
  * ``get_cached(name, token_pt=None) -> str | None`` — sync, disk-only.
  * ``fetch_async(name, cb, token_pt=None)`` — enqueue; ``cb(name, path|None)``
    is invoked on the worker thread (or ~immediately if already cached/missed).
  * ``prefetch(names, progress_cb=None) -> dict`` — blocking batch (CLI use).

CLI::

    python train/scryfall_cache.py "Lightning Bolt" "Thalia, Guardian of Thraben"
    python train/scryfall_cache.py --prefetch-vocab
    python train/scryfall_cache.py --token "Warrior" --pt 1/1
"""

import json
import os
import queue
import sys
import threading
import time

import requests

# Reuse the engine-mirroring uid derivation (Qt-free; decode only pulls numpy).
from decode import _name_to_uid

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve the repo root the same way decode.py locates bin/resources/cardsfolder
# (two dirs up from this file), then mirror into a sibling card_images/ dir.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "card_images")
_MISSES_PATH = os.path.join(_CACHE_DIR, "misses.json")

# ── HTTP politeness ───────────────────────────────────────────────────────────
_API = "https://api.scryfall.com"
_HEADERS = {"User-Agent": "robomage-gui/0.1", "Accept": "application/json"}
_TIMEOUT = 10.0
_MIN_INTERVAL = 0.120          # >=120 ms between requests (< ~8 req/s)

_throttle_lock = threading.Lock()
_last_request = 0.0            # monotonic clock of the previous HTTP request


def _throttle():
    """Block until at least _MIN_INTERVAL has elapsed since the last request."""
    global _last_request
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


# ── Miss manifest ─────────────────────────────────────────────────────────────
_misses_lock = threading.Lock()
_misses = None                 # cache-key -> ISO timestamp; lazily loaded


def _load_misses():
    global _misses
    if _misses is not None:
        return _misses
    with _misses_lock:
        if _misses is not None:
            return _misses
        data = {}
        try:
            with open(_MISSES_PATH, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
        _misses = data
    return _misses


def _record_miss(key):
    misses = _load_misses()
    with _misses_lock:
        misses[key] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            tmp = _MISSES_PATH + ".part"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(misses, fh, indent=0, sort_keys=True)
            os.replace(tmp, _MISSES_PATH)
        except OSError:
            pass


def _is_miss(key):
    return key in _load_misses()


# ── Cache-key / path helpers ──────────────────────────────────────────────────

def _cache_key(name, token_pt):
    """Cache key (and thus filename stem) for a card / token lookup.

    Named cards key on their uid; tokens fold P/T into the key so a 1/1 and a
    2/2 of the same token name are distinct files."""
    uid = _name_to_uid(name)
    if token_pt is None:
        return uid
    p, t = token_pt
    if p is None or t is None:
        return f"token_{uid}"
    return f"token_{uid}_{p}_{t}"


def _cache_path(key):
    return os.path.join(_CACHE_DIR, key + ".jpg")


# ── HTTP fetch primitives ─────────────────────────────────────────────────────

def _get_json(url, params=None):
    """GET JSON. Returns (status_code, data) on an HTTP response (data may be an
    error object), or (None, None) on a transient network error."""
    _throttle()
    try:
        resp = requests.get(url, params=params, headers=_HEADERS,
                            timeout=_TIMEOUT)
    except requests.exceptions.RequestException:
        return None, None
    try:
        data = resp.json()
    except ValueError:
        data = None
    return resp.status_code, data


def _get_bytes(url):
    """GET raw bytes (an image). Returns bytes on success, None on any failure."""
    _throttle()
    try:
        resp = requests.get(url, headers={"User-Agent": _HEADERS["User-Agent"]},
                            timeout=_TIMEOUT)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.content


def _image_url_from_card(card, face_name):
    """Extract the 'normal' image URL from a Scryfall card object.

    DFC/split/transform layouts carry no top-level image_uris; pick the
    card_faces[] entry whose name equals the requested face name
    (case-insensitive), else face 0."""
    uris = card.get("image_uris")
    if isinstance(uris, dict) and uris.get("normal"):
        return uris["normal"]
    faces = card.get("card_faces")
    if isinstance(faces, list) and faces:
        target = (face_name or "").strip().lower()
        chosen = None
        for face in faces:
            if not isinstance(face, dict):
                continue
            if target and face.get("name", "").strip().lower() == target \
                    and isinstance(face.get("image_uris"), dict):
                chosen = face
                break
        if chosen is None:
            chosen = faces[0] if isinstance(faces[0], dict) else None
        if chosen is not None:
            furis = chosen.get("image_uris")
            if isinstance(furis, dict) and furis.get("normal"):
                return furis["normal"]
    return None


def _lookup_named(name):
    """Resolve a named card's image URL. Returns (url, definitive_miss)."""
    status, data = _get_json(_API + "/cards/named", params={"exact": name})
    if status is None:                       # network error — transient
        return None, False
    if status == 404:
        status, data = _get_json(_API + "/cards/named", params={"fuzzy": name})
        if status is None:
            return None, False
    if status == 200 and isinstance(data, dict):
        url = _image_url_from_card(data, name)
        if url:
            return url, False
        return None, True                    # card found but no usable image
    if status == 404:
        return None, True                    # definitively not found
    return None, False                       # other HTTP error — treat transient


def _lookup_token(name, token_pt):
    """Resolve a token's image URL via /cards/search. Returns (url, miss)."""
    queries = []
    if token_pt is not None and token_pt[0] is not None and token_pt[1] is not None:
        p, t = token_pt
        queries.append(f'!"{name}" t:token pow:{p} tou:{t}')
    queries.append(f'!"{name}" t:token')
    saw_definitive_empty = False
    for q in queries:
        status, data = _get_json(_API + "/cards/search", params={"q": q})
        if status is None:                   # transient
            return None, False
        if status == 200 and isinstance(data, dict):
            results = data.get("data") or []
            if results:
                url = _image_url_from_card(results[0], name)
                if url:
                    return url, False
            else:
                saw_definitive_empty = True
        elif status == 404:
            saw_definitive_empty = True      # search matched nothing
        # any other status: fall through to the next (broader) query
    return None, saw_definitive_empty


def _do_fetch(name, token_pt):
    """Synchronous fetch-with-cache for one card/token. Throttled; safe to call
    from any thread. Returns the on-disk path, or None on failure (recording a
    definitive miss but not a transient network error)."""
    key = _cache_key(name, token_pt)
    path = _cache_path(key)
    if os.path.exists(path):
        return path
    if _is_miss(key):
        return None

    if token_pt is None:
        url, miss = _lookup_named(name)
    else:
        url, miss = _lookup_token(name, token_pt)

    if url is None:
        if miss:
            _record_miss(key)
        return None

    blob = _get_bytes(url)
    if not blob:
        return None                          # transient download failure

    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = os.path.join(_CACHE_DIR, key + ".part")
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, path)
    except OSError:
        return None
    return path


# ── Async download worker ─────────────────────────────────────────────────────
_work_q = queue.Queue()
_inflight = set()              # cache-keys already queued (dedup)
_inflight_lock = threading.Lock()
_worker = None
_worker_lock = threading.Lock()


def _ensure_worker():
    global _worker
    if _worker is not None:
        return
    with _worker_lock:
        if _worker is not None:
            return
        _worker = threading.Thread(target=_worker_loop, name="scryfall-dl",
                                   daemon=True)
        _worker.start()


def _worker_loop():
    while True:
        name, token_pt, cb, key = _work_q.get()
        try:
            path = _do_fetch(name, token_pt)
        except Exception:                    # never let the worker die
            path = None
        with _inflight_lock:
            _inflight.discard(key)
        if cb is not None:
            try:
                cb(name, path)
            except Exception:
                pass


# ── Public API ────────────────────────────────────────────────────────────────

def get_cached(name, token_pt=None):
    """Return the on-disk image path for a card/token if already downloaded,
    else None. Never touches the network."""
    path = _cache_path(_cache_key(name, token_pt))
    return path if os.path.exists(path) else None


def fetch_async(name, cb, token_pt=None):
    """Request a card/token image. ``cb(name, path_or_None)`` is invoked after the
    attempt — synchronously here if the image is already cached or a known miss,
    otherwise on the download worker thread once the fetch completes."""
    key = _cache_key(name, token_pt)
    path = _cache_path(key)
    if os.path.exists(path):
        if cb is not None:
            cb(name, path)
        return
    if _is_miss(key):
        if cb is not None:
            cb(name, None)
        return
    with _inflight_lock:
        if key in _inflight:
            return                           # already queued; its cb will fire
        _inflight.add(key)
    _ensure_worker()
    _work_q.put((name, token_pt, cb, key))


def prefetch(names, progress_cb=None):
    """Blocking batch fetch (used by the CLI). Returns {name: path|None}.
    ``progress_cb(i, total, name, path)`` is called after each item if given."""
    results = {}
    total = len(names)
    for i, name in enumerate(names, 1):
        path = _do_fetch(name, None)
        results[name] = path
        if progress_cb is not None:
            progress_cb(i, total, name, path)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli(argv):
    if "--prefetch-vocab" in argv:
        import card_costs
        names = [n for n in card_costs._VOCAB_NAMES if n and n.strip()]
        # De-dupe while preserving order (front/back faces are distinct names).
        seen, uniq = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                uniq.append(n)

        def _progress(i, total, name, path):
            mark = path if path else "MISS"
            print(f"[{i}/{total}] {name} -> {mark}")

        print(f"Warming {len(uniq)} vocab cards into {_CACHE_DIR} ...")
        prefetch(uniq, _progress)
        return 0

    if "--token" in argv:
        i = argv.index("--token")
        if i + 1 >= len(argv):
            print("usage: --token \"Name\" [--pt P/T]")
            return 2
        name = argv[i + 1]
        token_pt = (None, None)          # a token with unknown P/T (key token_<uid>)
        if "--pt" in argv:
            j = argv.index("--pt")
            if j + 1 < len(argv) and "/" in argv[j + 1]:
                p, t = argv[j + 1].split("/", 1)
                token_pt = (p.strip(), t.strip())
        cached = get_cached(name, token_pt)
        marker = "cached" if cached else "fetch"
        path = cached or _do_fetch(name, token_pt)
        print(f"{name} [{marker}] -> {path or 'MISS'}")
        return 0 if path else 1

    names = [a for a in argv if not a.startswith("--")]
    if not names:
        print(__doc__)
        return 0
    rc = 0
    for name in names:
        cached = get_cached(name)
        marker = "cached" if cached else "fetch"
        path = cached or _do_fetch(name, None)
        print(f"{name} [{marker}] -> {path or 'MISS'}")
        if not path:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
