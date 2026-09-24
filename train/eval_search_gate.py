if __name__ == "__main__":
    import sys
    sys.exit("eval_search_gate.py was removed; use `train.py baseline --player-a "
             "mcts:gen --player-b gen --deck-a D --games N --sims 128 --worlds 4 "
             "--workers W` (search vs the raw policy on a mirror; the scripted:hard "
             "reference legs are baseline runs with the default --player-b)")
