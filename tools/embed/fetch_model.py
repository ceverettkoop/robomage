#!/usr/bin/env python3
"""Download the pinned text-embedding model's files (plan step 5c, docs/featurizer.md §5).

Standalone tool, Python standard library only. Reads MODEL_PIN (next to this script) and
fetches the files the generator bin/<config>/rmembed reads, at the pinned commit, into
<cache>/<repo with / as __>/<commit>/ (default cache: .cache/embed/ at the repository root):

    config.json  tokenizer_config.json  vocab.txt  model.safetensors

A file already present with the right size is kept. Prints the model directory on stdout and
exits 0; on any failure prints why on stderr and exits 1.

    python3 tools/embed/fetch_model.py [--cache DIR] [--pin FILE]
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
HF = "https://huggingface.co"
FILES = ["config.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]


def read_pin(path):
    """The pin's key/value lines (`key value`, `#` comments), as a dict."""
    pin = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            pin[key] = value.strip()
    for key in ("repo", "commit"):
        if not pin.get(key):
            raise ValueError("%s: missing '%s' line" % (path, key))
    return pin


def remote_sizes(repo, commit):
    """Each file's size at `commit`, from the Hugging Face tree API (LFS files report their
    object size)."""
    url = "%s/api/models/%s/tree/%s?recursive=true" % (HF, repo, commit)
    with urllib.request.urlopen(url, timeout=60) as r:
        entries = json.load(r)
    sizes = {}
    for e in entries:
        lfs = e.get("lfs")
        sizes[e["path"]] = lfs["size"] if lfs else e.get("size")
    return sizes


def fetch(repo, commit, name, dest):
    url = "%s/%s/resolve/%s/%s" % (HF, repo, commit, name)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", default=os.path.join(HERE, "MODEL_PIN"))
    ap.add_argument("--cache", default=os.path.join(REPO_ROOT, ".cache", "embed"))
    args = ap.parse_args()
    try:
        pin = read_pin(args.pin)
        repo, commit = pin["repo"], pin["commit"]
        model_dir = os.path.join(args.cache, repo.replace("/", "__"), commit)
        os.makedirs(model_dir, exist_ok=True)
        sizes = remote_sizes(repo, commit)
        for name in FILES:
            if name not in sizes:
                raise ValueError("%s@%s has no %s" % (repo, commit, name))
            dest = os.path.join(model_dir, name)
            if os.path.isfile(dest) and os.path.getsize(dest) == sizes[name]:
                continue
            print("fetch_model: downloading %s/%s" % (repo, name), file=sys.stderr)
            fetch(repo, commit, name, dest)
            if os.path.getsize(dest) != sizes[name]:
                raise ValueError("%s: got %d bytes, expected %d" % (name, os.path.getsize(dest), sizes[name]))
    except (OSError, ValueError, urllib.error.URLError) as e:
        print("fetch_model: %s" % e, file=sys.stderr)
        print("fetch_model: the generator needs network access to huggingface.co and its CDN", file=sys.stderr)
        return 1
    print(model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
