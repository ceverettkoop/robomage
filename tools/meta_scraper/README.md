# meta_scraper

Standalone scraper that pulls the current top Legacy metagame decks from
[MTGTop8](https://mtgtop8.com) and writes them as robomage `.dk` deck files.

It depends **only on the Python standard library** and imports nothing from the
robomage engine or `train/`, so it can run with any Python 3.

## Usage

```bash
# Preview the top 10 archetypes without writing files
python tools/meta_scraper/scrape_meta.py --top 10 --dry-run

# Write them into bin/resources/decks/meta/
python tools/meta_scraper/scrape_meta.py --top 10
```

Options:

| flag | default | meaning |
|---|---|---|
| `--top N` | 10 | number of top archetypes to fetch |
| `--out DIR` | `bin/resources/decks/meta/` | output directory for `.dk` files |
| `--format CODE` | `LE` | MTGTop8 format code (LE = Legacy) |
| `--delay SEC` | 0.6 | pause between requests (be polite) |
| `--dry-run` | off | print the plan without writing files |

The run is idempotent: files are overwritten by sanitized archetype name.

## How it works

1. **Metagame page** `https://mtgtop8.com/format?f=LE` lists archetypes in
   descending meta-share order as `archetype?a=ID` links, each with a `… %`.
2. **Archetype page** `https://mtgtop8.com/archetype?a=ID&f=LE` lists recent
   decklists as `event?e=E&d=DECKID` rows; the first row is taken as the
   representative list.
3. **Deck export** `https://mtgtop8.com/mtgo?d=DECKID` returns plain MTGO text
   (`<qty> <name>` lines + a `Sideboard` separator), converted to the `.dk`
   format used elsewhere in `bin/resources/decks/` (tab-separated `qty<TAB>name`,
   a blank line, then `SIDEBOARD:`).

Card names are kept verbatim (apostrophes included); the engine's `name_to_uid`
normalization strips punctuation when resolving card scripts, so both spellings
resolve.

## Notes

- Some archetypes are 80-card **companion** (e.g. Yorion) builds — an 80-card
  mainboard is expected for those, not a parsing error.
- Card names that are not yet in `src/card_vocab.h` are still written to the
  deck file; use `train/missing_cards.py` to list which ones need implementing.
- If MTGTop8 changes its HTML and parsing returns nothing, the tool exits with a
  clear error rather than writing empty decks.
