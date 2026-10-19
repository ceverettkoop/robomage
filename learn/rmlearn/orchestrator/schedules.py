"""The actor schedules of the phases (docs/actor.md §2; docs/rmtrain.md §6): the teacher corpus (B0),
DAgger rounds (B2) and R-NaD self-play (B4, B5). Each is a dict the orchestrator writes whole to the
run's ``schedule.json``; its ``version`` names the phase (and round), and the actors' results lines
carry it, so the orchestrator counts each phase's games from them."""

import itertools
import os

from ..corpus import teacher_command



def league_pairs(cfg):
    """[(deck_a, deck_b, weight)] of the league: league.pairs, or every pair of league.decks."""
    lg = cfg["league"]
    if lg["pairs"]:
        return [(p[0], p[1], float(p[2]) if len(p) > 2 else 1.0) for p in lg["pairs"]]
    return [(a, b, 1.0) for a, b in itertools.combinations(lg["decks"], 2)]


def deck(stem):
    return stem if "/" in stem else "league/" + stem


def nn_remote(sock, policies, options=""):
    spec = "nn:remote:%s:policies=%s" % (sock, policies)
    return spec + (":" + options.strip(":") if options else "")


def _defaults(cfg, omniscient):
    lg = cfg["league"]
    return {"format": lg["format"], "knowledge_a": lg["knowledge"], "knowledge_b": lg["knowledge"],
            "max_game_decisions": lg["max_game_decisions"], "trajectory_omniscient": bool(omniscient)}


def _outputs(run, phase, episodes=True):
    out = {"rmlog_dir": run.rmlog_dir(phase)}
    out["trajectory_dir"] = run.episode_dir(phase) if episodes else None
    for d in out.values():
        if d:
            os.makedirs(d, exist_ok=True)
    return out


def teacher(cfg):
    return teacher_command(cfg["actors"]["teacher_tier"], cfg["run"]["seed"])


def b0(run, cfg):
    """Scripted:hard vs scripted:hard over the league (both seats ``first`` + the teacher at mix 1,
    swap decks), and with share b0.explore_frac scripted vs explore (the scripted side recorded)."""
    pairs = league_pairs(cfg)
    total = sum(w for _, _, w in pairs) or 1.0
    frac = cfg["b0"]["explore_frac"]
    t = teacher(cfg)
    entries = []
    for a, b, w in pairs:
        if w <= 0:
            continue
        entries.append({"id": "ss-%s-%s" % (a, b), "weight": (1 - frac) * w / total, "deck_a": deck(a),
                        "deck_b": deck(b), "swap": "decks", "agent_a": "first", "agent_b": "first", "teacher_a": t,
                        "teacher_b": t, "teacher_mix": 1})
        if frac > 0:
            for x, y in ((a, b), (b, a)):
                entries.append({"id": "se-%s-%s" % (x, y), "weight": frac * w / total / 2, "deck_a": deck(x),
                                "deck_b": deck(y), "swap": "sides", "agent_a": "first", "agent_b": "explore",
                                "teacher_a": t, "teacher_mix": 1, "trajectory_seats": "A"})
    return {"version": "b0", "outputs": {"train": _outputs(run, "b0")},
            "defaults": _defaults(cfg, cfg["league"]["omniscient"]), "entries": entries}


def dagger(run, cfg, round_no, mix):
    """The current BC policy (server policy 0, sampled) on both seats with the teacher labelling every
    decision of both, its choice played with probability `mix`."""
    t = teacher(cfg)
    agent = nn_remote(run.rmip_sock, "0", cfg["b2"]["nn_options"])
    entries = []
    for a, b, w in league_pairs(cfg):
        if w <= 0:
            continue
        e = {"id": "dg-%s-%s" % (a, b), "weight": w, "deck_a": deck(a), "deck_b": deck(b), "swap": "decks",
             "agent_a": agent, "agent_b": agent, "teacher_a": t, "teacher_b": t}
        if mix > 0:
            e["teacher_mix"] = float(mix)
        entries.append(e)
    return {"version": "b2-r%d" % round_no, "outputs": {"train": _outputs(run, "dagger")},
            "defaults": _defaults(cfg, cfg["league"]["omniscient"]), "entries": entries}


def selfplay(run, cfg, version):
    """R-NaD self-play: policies 0, 1, 2 evaluated at every decision (the regularisation policies'
    log π recorded), sampled from 0, on both seats."""
    agent = nn_remote(run.rmip_sock, "0,1,2", cfg["b4"]["nn_options"])
    entries = []
    for a, b, w in league_pairs(cfg):
        if w <= 0:
            continue
        entries.append({"id": "sp-%s-%s" % (a, b), "weight": w, "deck_a": deck(a), "deck_b": deck(b),
                        "swap": "decks", "agent_a": agent, "agent_b": agent})
    return {"version": version, "outputs": {"train": _outputs(run, "selfplay")},
            "defaults": _defaults(cfg, cfg["model"]["omniscient_critic"]), "entries": entries}
