# tools/scryfall — card images for the workbench

`fetch_images.py` fills the workbench's card-image cache from the [Scryfall API](https://scryfall.com/docs/api).
Python standard library only (`python3`, no venv); it imports nothing from the engine. The workbench
(`tools/rmwork`, `docs/workbench_plan.md` §2.4) reads only the cache and never touches the network, so
without a cache every card is drawn as text.

```bash
python3 tools/scryfall/fetch_images.py                      # every vocab card and token (~15 minutes)
python3 tools/scryfall/fetch_images.py --cards "Lightning Bolt" "token:c_a_clue_draw"
python3 tools/scryfall/fetch_images.py --printing "Lightning Bolt=m10:146"   # one override's images
python3 tools/scryfall/fetch_images.py --dry-run            # list and choose, write nothing
python3 tools/scryfall/fetch_images.py --selftest           # the choice rule on canned JSON, no network
```

## What it fetches

- **Cards:** every vocab index of `src/card_vocab.h` (aliases of one index, such as a split card's
  halves, are one card, keyed by its `Front/Back` name). Looked up by exact name, then fuzzy; then
  every paper printing of its oracle card (`oracleid:… game:paper`, `unique=prints`).
- **Tokens:** every token script under `bin/resources/tokenscripts/` (what
  `tools/forge_fetch/provision_decks.py` provisions for the decks), keyed `token:<stem>`, with its
  vocab index when the token band registers it. Searched as `t:token !"<name>"` with the script's
  colours, artifact / creature / planeswalker / legendary types, power / toughness and keywords; the
  keywords, then the power / toughness, then the colours are dropped until a token matches.

## The default printing

The **oldest** printing (released date, then set code, then collector number in natural order)
that is printed on paper and not digital, not a promo, not oversized, not from a `memorabilia`,
`funny`, `minigame`, `alchemy`, `vanguard`, `planechase`, `archenemy` or `treasure_chest` set (gold
borders, Collectors' Edition, un-sets) and has an image on Scryfall. When no printing passes, the
oldest paper printing with an image. Its `normal` image (488 × 680 JPEG) is downloaded; a double-faced
card's two faces each.

A **token** is chosen by the same rule, except that a printing from the token set of a card that makes
it comes first: the vocab cards whose scripts make the token (`bin/resources/cardsfolder/`, scanned as
`tools/forge_fetch/provision_decks.py` does: `TokenScript$` stems, Amass armies, Investigate's Clue, …)
each give the set of their own default printing, and Scryfall prints set `S`'s tokens in set `t<S>`,
so Orcish Bowmasters (`ltr`) gets its Orc Army from `tltr` and Tamiyo, Inquisitive Student (`mh3`) her
Clue from `tmh3` rather than the oldest Clue (`tsoi`). The oldest such printing that may be the default
wins; when no maker's set has one (or no script makes the token, or its maker is not listed yet), the
oldest as for a card. The manifest keeps a token's makers and their sets as `sources`; a token listed
before this rule (no `sources`) is listed again by the next run.

The workbench lists every paper printing (promos too) in Edit → Settings →
Card Printings, where a card's printing can be changed (`~/.config/robomage/printings.json`); a chosen
printing whose image is not cached is fetched with `--printing`, which the tab runs for you.

## The cache

Default `$XDG_CACHE_HOME/robomage/scryfall` (else `~/.cache/robomage/scryfall`); `--cache DIR`
elsewhere (Settings → Card Printings' "Image cache" must then name it).

```
manifest.json               per card: key, vocab index, names, every paper printing (set, set name,
                            number, date, set type, promo, faces: name, URL, file) and the default;
                            a token's sources (the cards that make it, their sets)
images/<set>/<number>.jpg   a printing's image (a double-faced card: <number>-0.jpg, <number>-1.jpg)
```

The run is resumable: a card already in the manifest is not listed again (unless named with
`--cards`, or with `--refresh`), and an image already cached is not downloaded again. Cards and tokens
Scryfall has nothing for are listed under the manifest's `missing` and at the end of the run.

## Scryfall's rules

A generic `User-Agent` (`robomage-workbench/1.0`) and `Accept` header; API requests 500 ms apart (its
search and named-card endpoints allow two a second, and a 429 costs a minute) and image downloads from
its CDN 100 ms apart; a retry with backoff on 429 and 5xx, honouring `Retry-After`. Card images are
© Wizards of the Coast; Scryfall serves them for personal, non-commercial use.
