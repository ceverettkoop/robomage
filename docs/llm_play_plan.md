# Plan: an LLM playing a seat against the model, over MCP

**Status: deferred (2026-11-08).** Written down, not started. Nothing here waits on a decision until it is
picked up; the open choices are at the end.

Goal: an LLM plays one seat of a game (or a bo3 match) against a learned policy, through an MCP server,
seeing only what that seat sees, in a concise text form, so that games can be played interactively (a
person or Claude Code driving the tools) and benchmarked (many games on fixed seeds against the model).

## 1. What exists

- **The model as an opponent.** `robomage_nn --player A --agent-b nn:<export>[:argmax]` is a human seat
  against a learned policy, evaluated in process on the CPU (`src/actor/nn_agent.h`): no GPU lock, so it
  can run beside a training run (CPU contention aside).
- **A concise, leak-free view.** The human seat (`src/cli/human.h`, rendered by `src/cli/board.h`) reads
  only its seat's decoded frames, knowledge tables included. At each decision it prints what the seat saw
  happen since its previous decision, the life and zone counts, the board only when it changed, the
  pending decision and a numbered menu (`tests/cli/human_vs_explore.out`). A decision is about 150–400
  tokens.
- **Detail on demand.** `card <name|#handle>`, `zone <name> [opp]`, `know` (the knowledge panel: what the
  opponent's hidden hand can hold), `log [n|all]`, `board`.
- **Auto-pass.** `z` / `z mine` answers plain priority decisions until the next turn (or the seat's own
  next turn), stopping at any other decision and when the opponent puts something on the stack.
- **Recording.** Every run writes an RMLOG decision log that replays exactly; `GAME_RESULT` /
  `MATCH_RESULT` lines give the score.
- **Prompting.** The seat prints `Player A [0-4, help]> `, flushes stdout and reads one line from stdin
  (`src/cli/human.cpp`). An unknown command or out-of-range row re-prompts; end of input or `quit` ends
  the run with `AGENT_QUIT` logged.

## 2. Work

1. **A prompt sentinel** (C++). A flag `--human-sentinel`: after each prompt the human seat prints one
   machine-matchable line, and one at the match's end, e.g.

   ```
   @@DECISION rows=0-4 kind=priority game=1 turn=3
   @@END result=B match=1-2
   ```

   Scraping the prompt text would work, but a sentinel doesn't break when the text format changes. Add
   it to the flag table (`src/cli_args.cpp`) and `cli_print_help`, and give the `cli` tier a golden
   session with it on.
2. **The MCP server**, `tools/mcp_play/server.py`, Python stdlib only. MCP's stdio transport is
   newline-delimited JSON-RPC 2.0 (`initialize`, `tools/list`, `tools/call`), so it needs no SDK. One
   engine subprocess per game; commands written to its stdin, stdout read to the next sentinel line.
   Tools:

   | Tool | Does |
   |---|---|
   | `new_game(deck, opp_deck, opponent="nn:<export>", format="bo1"/"bo3", seed)` | starts the engine (cwd `bin/`), returns the first decision |
   | `choose(row)` | answers with a menu row; returns everything up to the next decision, or the result |
   | `pass_turn(mine=false)` | `z` / `z mine`: what happened meanwhile, then the next decision |
   | `look(what)` | `board`, `zone hand opp`, `card Lightning Bolt`, `know`, `log 20`; doesn't advance the game |
   | `concede(match=false)` | concedes the game or the match |

   Each result is the CLI text plus a small structured part: the menu as `[{row, text}]`, game, turn, life
   totals, game over or not, and the decision log's path. A re-prompt (bad row, unknown command) comes
   back as a tool error, so a bad choice can't break the game. Opponents: any in-process agent:
   `nn:<export>[:argmax|:temp=…]`, `explore`, `random`. `nn:` needs `robomage_nn`; a history export needs
   `--feature-history N`, as in `rmtrain bench`. The scripted agent isn't in-process, since it plays
   through `python/rm_driver.py`, so it isn't offered.
3. **Conciseness**, measured on a first real game before changing anything. Likely candidates:
   - shorter rows for the opponent's battlefield;
   - graveyards as a count plus their last few cards;
   - the full board every few decisions, with deltas in between.

   These would come as a `--cli-compact` option so the human CLI's goldens don't move. Card text stays
   behind `card`: an LLM knows most cards by name. The exception is where the engine deviates from the
   rules (`docs/rules_deviations.md`, e.g. the name-a-card candidate set), which the server's tool
   descriptions or an MCP prompt should mention.
4. **A check.** A stdlib client drives the server against `explore` through a bo1 and a bo3 game: start,
   choose, pass, look, concede and a finished game. Each game's decision log must replay to the same
   result. It runs on the plain engine with no torch, in `make check`'s `cli` or `smoke` tier. An `nn:`
   opponent case goes in the opt-in `actor` tier.
5. **Connecting it and evaluating.**
   - **Claude Code:** `claude mcp add robomage -- python3 tools/mcp_play/server.py`, or a project
     `.mcp.json`. The user makes that config change.
   - **Benchmark:** N games on fixed seeds and deck pairs, scored like `rmtrain bench` (per-game win
     rate, game 1 apart from games 2–3), so "LLM vs final" sits beside `scripted:hard` vs final on the
     same matches.
   - **Driving many games:** at about 100–200 tool calls a game, a small API harness does it better
     than an interactive session.

## 3. Risks

- **Cost and speed.** About 30–80k input tokens a game even with auto-pass; step 3 is where that
  comes down.
- **Multi-step sequences.** Attack and block declarations and manual payments are several menus in a
  row. Auto-pass doesn't cover them by design, but each step is a short menu.
- **Leaks.** Everything comes from the human seat's own frames, so the LLM sees exactly what a player in
  that seat sees. The opponent's decklist is open: `card <name>` reaches any card of the match, as a
  person at the CLI can.
- **The per-game decision cap** ends a game that runs too long: end reason `cap`, no result.

## 4. Open choices

- **Who plays first:** interactive (a person, or Claude Code through the tools), or an API harness over
  many games for a benchmark.
- **Format:** bo1 to start, or bo3 with sideboarding from the start (the human seat already handles the
  sideboarding menus).
