"""
Self-play PPO training for RoboMage.

Dependencies:
    pip install gymnasium stable-baselines3 sb3-contrib

sb3-contrib provides MaskablePPO, which respects the action_masks() method
from the environment so the agent never wastes probability mass on illegal actions.

Usage:
    # From the robomage repo root:
    cd train
    python train.py

    # Override binary path:
    python train.py --binary ../bin/robomage

    # Resume from checkpoint:
    python train.py --load checkpoints/robomage_100000_steps.zip
"""

import argparse
import os

from env import RoboMageEnv, ModelVsScriptedEnv, SelfPlayEnv, NarrativeEnv, scripted_action, OBS_SIZE, STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, BINARY
from extractor import CardGameExtractor

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    USE_MASKABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    USE_MASKABLE = False
    print("Warning: sb3-contrib not found, using plain PPO (no action masking).")
    print("Install with: pip install sb3-contrib")

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


class WinTallyCallback(BaseCallback):
    """Prints win rate since the last rollout after each rollout."""

    def __init__(self):
        super().__init__()
        self._interval_model_wins = 0
        self._interval_scripted_wins = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            r = info["episode"]["r"]
            if r > 0:
                self._interval_model_wins += 1
            elif r < 0:
                self._interval_scripted_wins += 1
        return True

    def _on_rollout_end(self) -> None:
        total = self._interval_model_wins + self._interval_scripted_wins
        if total == 0:
            return
        pct = 100.0 * self._interval_model_wins / total
        print(f"[tally] model wins: {self._interval_model_wins}  scripted wins: {self._interval_scripted_wins}  "
              f"total: {total}  model win rate: {pct:.1f}%")
        self._interval_model_wins = 0
        self._interval_scripted_wins = 0


class ReplayLogCallback(BaseCallback):
    """After each rollout, runs one model-vs-scripted game and saves a transcript."""

    def __init__(self, binary_path: str, replay_dir: str = "replays"):
        super().__init__()
        self.binary_path = binary_path
        self.replay_dir = replay_dir
        self._rollout = 0
        os.makedirs(replay_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import numpy as np
        self._rollout += 1
        log_path = os.path.join(self.replay_dir, f"rollout_{self._rollout:05d}.txt")

        env = NarrativeEnv(binary_path=self.binary_path)
        if USE_MASKABLE:
            from sb3_contrib.common.wrappers import ActionMasker as _AM
            masked = _AM(env, lambda e: e.action_masks())
        else:
            masked = env

        try:
            obs, _ = masked.reset()
            model_is_a = bool(np.random.random() < 0.5)
            done = False
            total_reward = 0.0
            turn = 0
            prev_active_is_a = None
            known_hand = {"A": [], "B": []}

            with open(log_path, "w") as f:
                model_side = "A" if model_is_a else "B"
                scripted_side = "B" if model_is_a else "A"
                f.write(f"=== Rollout {self._rollout}: Model ({model_side}) vs Scripted ({scripted_side}) ===\n\n")

                while not done:
                    for line in env.flush_lines():
                        if line.strip():
                            f.write(line + "\n")

                    a_has_priority = obs[31] > 0.5
                    model_has_priority = a_has_priority if model_is_a else not a_has_priority
                    num_choices = env._num_choices
                    cur_side = "A" if a_has_priority else "B"

                    priority_is_a = a_has_priority
                    active_is_a = (obs[30] > 0.5) == priority_is_a
                    cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
                    is_mulligan = any(c == 11 for c in cats)

                    known_hand[cur_side] = _decode_hand(obs)

                    if not is_mulligan and active_is_a != prev_active_is_a:
                        turn += 1
                        active_label = "A" if active_is_a else "B"
                        f.write(f"--- Turn {turn} (Player {active_label}) ---\n")
                        f.write(f"  PA: {', '.join(known_hand['A']) or '(empty)'}\n")
                        f.write(f"  PB: {', '.join(known_hand['B']) or '(empty)'}\n")
                        prev_active_is_a = active_is_a

                    if model_has_priority:
                        masks = env.action_masks() if USE_MASKABLE else None
                        action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
                        action = int(action)
                        f.write(f"[Model/{cur_side}] chose {action} of {num_choices}\n")
                    else:
                        action = scripted_action(obs, num_choices)
                        f.write(f"[Scripted/{cur_side}] chose {action} of {num_choices}\n")

                    obs, reward, terminated, truncated, _ = masked.step(action)
                    total_reward += reward
                    done = terminated or truncated

                for line in env.flush_lines():
                    if line.strip():
                        f.write(line + "\n")

                model_reward = total_reward if model_is_a else -total_reward
                result = "Model wins" if model_reward > 0 else "Scripted wins" if model_reward < 0 else "Draw"
                f.write(f"\n=== {result} ===\n")

            print(f"[replay] rollout {self._rollout}: {result} -> {log_path}")
        except Exception as exc:
            print(f"[replay] rollout {self._rollout}: game failed ({exc})")
        finally:
            env.close()


CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 10  # parallel game processes


def make_env(binary_path: str, rank: int):
    def _init():
        env = ModelVsScriptedEnv(binary_path=binary_path)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def make_self_play_env(binary_path: str, checkpoint_dir: str, rank: int):
    def _init():
        env = SelfPlayEnv(checkpoint_dir=checkpoint_dir, binary_path=binary_path)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def train(binary_path: str, load_path: str | None = None, total_timesteps: int = TOTAL_TIMESTEPS,
          tally: bool = False, self_play: bool = False, scripted_fraction: float = 0.0):
    """Train the model.

    ``scripted_fraction`` controls how many of the N_ENVS parallel environments
    use the scripted agent instead of self-play.  E.g. 0.0 = all self-play,
    0.3 = ~2 scripted + 5 self-play (with N_ENVS=7).  Has no effect unless
    ``self_play`` is also True.  Mixing in scripted environments prevents the
    policy drifting to strategies that beat itself but lose to general play.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Parallel environments for faster data collection
    if self_play:
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
        n_scripted = round(N_ENVS * scripted_fraction)
        n_self_play = N_ENVS - n_scripted
        env_fns = (
            [make_self_play_env(binary_path, checkpoint_dir, i) for i in range(n_self_play)]
            + [make_env(binary_path, N_ENVS - n_scripted + i) for i in range(n_scripted)]
        )
        if n_scripted:
            print(f"Env mix: {n_self_play} self-play + {n_scripted} scripted")
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv([make_env(binary_path, i) for i in range(N_ENVS)])

    policy_kwargs = dict(
        features_extractor_class=CardGameExtractor,
        net_arch=[256, 256],
    )

    if load_path:
        print(f"Resuming from {load_path}")
        model = MaskablePPO.load(load_path, env=vec_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,           # steps per env per update
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,         # encourage exploration early on
            verbose=1,
            tensorboard_log=LOG_DIR,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=25_000 // N_ENVS,
            save_path=CHECKPOINT_DIR,
            name_prefix="robomage",
        ),
    ]
    if tally:
        callbacks.append(WinTallyCallback())
    callbacks.append(ReplayLogCallback(binary_path=binary_path))

    print(f"Training for {total_timesteps:,} timesteps across {N_ENVS} envs...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=load_path is None)
    model.save(os.path.join(CHECKPOINT_DIR, "robomage_final"))
    print("Saved final model.")

    vec_env.close()


def diag(binary_path: str, n_games: int = 10):
    """Quick diagnostic: spin up a fresh random model and run n_games vs scripted.

    Logs every decision (like watch_scripted).  On a draw the full log is saved
    to diag_draw_<game>.txt and printed to stdout, since draws should not occur.
    """
    import numpy as np

    policy_kwargs = dict(features_extractor_class=CardGameExtractor, net_arch=[256, 256])

    # Create a throw-away env just to give MaskablePPO the spaces it needs.
    _tmp_env = NarrativeEnv(binary_path=binary_path)
    if USE_MASKABLE:
        _tmp_env = ActionMasker(_tmp_env, lambda e: e.action_masks())
    model = MaskablePPO("MlpPolicy", _tmp_env, policy_kwargs=policy_kwargs, verbose=0)
    _tmp_env.close()

    wins = losses = draws = 0
    print(f"Running {n_games} games (random model vs scripted)...")
    for i in range(n_games):
        env = NarrativeEnv(binary_path=binary_path)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        decision = 0
        turn = 0
        prev_active_is_a = None
        known_hand = {"A": [], "B": []}
        log_lines = [f"=== Game {i+1}: Random Model (A) vs Scripted (B) ===\n"]

        while not done:
            # Flush narrative lines from the game process
            raw_env = env.env if hasattr(env, "env") else env
            for line in raw_env.flush_lines():
                if line.strip():
                    log_lines.append(line)

            num_choices = raw_env._num_choices
            priority_is_a = obs[31] > 0.5
            player = "A" if priority_is_a else "B"
            active_is_a = (obs[30] > 0.5) == priority_is_a
            step_name = _STEP_NAMES[int(np.argmax(obs[18:30]))]

            cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
            is_mulligan = any(c == 11 for c in cats)

            known_hand[player] = _decode_hand(obs)

            if not is_mulligan and active_is_a != prev_active_is_a:
                turn += 1
                active_label = "A" if active_is_a else "B"
                a_hand = ", ".join(known_hand["A"]) or "(empty)"
                b_hand = ", ".join(known_hand["B"]) or "(empty)"
                log_lines.append(f"--- Turn {turn} (Player {active_label}) ---")
                log_lines.append(f"  PA: {a_hand}")
                log_lines.append(f"  PB: {b_hand}")
                prev_active_is_a = active_is_a

            # Model controls player A, scripted controls player B
            if priority_is_a:
                masks = raw_env.action_masks() if USE_MASKABLE else None
                action, _ = model.predict(obs, action_masks=masks, deterministic=False)
                action = int(action)
                agent_label = "Model/A"
            else:
                action = scripted_action(obs, num_choices)
                agent_label = "Scripted/B"

            chosen_cat = _CAT_NAMES.get(int(cats[action]), str(cats[action]))
            all_cats = [_CAT_NAMES.get(int(c), str(c)) for c in cats]
            log_lines.append(
                f"[{decision:4d}] P{player}  {step_name:<14}  choices={num_choices}"
                f"  available={all_cats}  -> {action} ({chosen_cat})  [{agent_label}]"
            )

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            decision += 1

        for line in raw_env.flush_lines():
            if line.strip():
                log_lines.append(line)

        env.close()

        if total_reward > 0:
            wins += 1
            result = "W"
        elif total_reward < 0:
            losses += 1
            result = "L"
        else:
            draws += 1
            result = "D"
            # Draws should not occur — save and print the full log
            log_path = f"diag_draw_{i+1}.txt"
            log_lines.append("\n=== DRAW (should not occur) ===")
            log_text = "\n".join(log_lines)
            with open(log_path, "w") as f:
                f.write(log_text + "\n")
            print(f"\n{'='*60}")
            print(f"DRAW in game {i+1} — saving log to {log_path}")
            print('='*60)
            print(log_text, flush=True)

        print(f"  game {i+1:2d}/{n_games}: {result}  (W:{wins} L:{losses} D:{draws})", flush=True)

    total = wins + losses + draws
    win_pct = 100 * wins / total if total else 0
    print(f"\n{wins}W / {losses}L / {draws}D over {n_games} games ({win_pct:.1f}% win rate)")


def baseline(binary_path: str, model_path: str, n_games: int = 100):
    """Evaluate win rate of the model against the scripted agent.

    The model is randomly assigned to Player A or B each game (matching training
    conditions).  Reward is from the model's perspective so wins/losses are
    counted directly.
    """
    import numpy as np
    model = MaskablePPO.load(model_path)
    env = ModelVsScriptedEnv(binary_path=binary_path)
    if USE_MASKABLE:
        env = ActionMasker(env, lambda e: e.action_masks())
    wins = losses = draws = 0

    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1
        print(f"\rGame {i+1}/{n_games}  W:{wins} L:{losses} D:{draws}", end="", flush=True)

    env.close()
    print()
    print(f"vs scripted over {n_games} games: {wins}W / {losses}L / {draws}D "
          f"({100 * wins / n_games:.1f}% win rate)")


def observe(binary_path: str, model_path: str):
    """Watch the model play one game against the scripted agent.

    The model is randomly assigned to Player A or B.  Both players' decisions
    are printed so the full game flow is visible.
    """
    import numpy as np
    model = MaskablePPO.load(model_path)
    env = RoboMageEnv(binary_path=binary_path, render_mode="human")
    obs, _ = env.reset()

    model_is_a = bool(np.random.random() < 0.5)
    model_side = "A" if model_is_a else "B"
    scripted_side = "B" if model_is_a else "A"
    print(f"=== Model (Player {model_side}) vs Scripted (Player {scripted_side}) ===\n")

    done = False
    total_reward = 0.0
    while not done:
        a_has_priority = obs[31] > 0.5
        model_has_priority = a_has_priority if model_is_a else not a_has_priority
        cur_side = "A" if a_has_priority else "B"
        num_choices = env._num_choices

        if model_has_priority:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)
            print(f"  [Model/{cur_side}]    action {action} of {num_choices}")
        else:
            action = scripted_action(obs, num_choices)
            print(f"  [Scripted/{cur_side}] action {action} of {num_choices}")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    print()
    model_reward = total_reward if model_is_a else -total_reward
    if model_reward > 0:
        print(f"=== Model ({model_side}) wins ===")
    elif model_reward < 0:
        print(f"=== Scripted ({scripted_side}) wins ===")
    else:
        print("=== Draw ===")


_CAT_NAMES = {
    0: "PASS", 1: "MANA", 2: "SEL_ATK", 3: "CONF_ATK",
    4: "SEL_BLK", 5: "CONF_BLK", 6: "ACTIVATE", 7: "CAST",
    8: "TARGET", 9: "LAND", 10: "OTHER", 11: "MULLIGAN", 12: "BOTTOM_CARD"
}

_STEP_NAMES = [
    "Untap", "Upkeep", "Draw", "First Main", "Begin Combat",
    "Declare Atk", "Declare Blk", "Combat Dmg",
    "End Combat", "Second Main", "End Step", "Cleanup",
]

# Index → card name, mirrors card_vocab.h
_VOCAB_NAMES = [
    "Mountain", "Forest", "Lightning Bolt", "Grizzly Bears", "Volcanic Island",
    "Scalding Tarn", "Flooded Strand", "Polluted Delta", "Wooded Foothills", "Misty Rainforest",
    "Wasteland", "Ponder", "Force of Will", "Daze", "Soul Warden", "Tundra",
    "Delver of Secrets", "Insectile Aberration", "Flying Men", "Island",
    "Dragon's Rage Channeler", "Air Elemental", "Counterspell", "Lightning Strike",
    "Brainstorm",
]


def _decode_hand(obs):
    """Return list of card names for the priority player's hand (obs[8557:8877])."""
    import numpy as np
    cards = []
    for slot in range(10):           # MAX_HAND_SLOTS = 10
        base = 8557 + slot * 32      # _HAND_START + slot * N_CARD_TYPES
        vec = obs[base : base + 32]
        idx = int(np.argmax(vec))
        if vec[idx] > 0.5:
            cards.append(_VOCAB_NAMES[idx] if idx < len(_VOCAB_NAMES) else f"?{idx}")
    return cards


def watch_scripted(binary_path: str):
    """Run one game with both players driven by the scripted agent and print every decision."""
    import numpy as np

    env = NarrativeEnv(binary_path=binary_path)
    obs, _ = env.reset()
    done = False
    decision = 0
    turn = 0
    prev_active_is_a = None       # None until first non-mulligan query
    known_hand = {"A": [], "B": []}

    print("=== Scripted (A) vs Scripted (B) ===\n", flush=True)

    while not done:
        # Print narrative from the previous step; skip blank lines (e.g. leading \n
        # from C++ printf("\n--- Declare Attackers...")) to avoid extra whitespace.
        for line in env.flush_lines():
            if line.strip():
                print(line, flush=True)

        num_choices = env._num_choices
        priority_is_a = obs[31] > 0.5
        player = "A" if priority_is_a else "B"
        # active_is_a: obs[30]=1 means priority player IS the active player
        active_is_a = (obs[30] > 0.5) == priority_is_a
        step_name = _STEP_NAMES[int(np.argmax(obs[18:30]))]

        cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
        is_mulligan = any(c == 11 for c in cats)

        # Cache the priority player's hand on every query
        known_hand[player] = _decode_hand(obs)

        # Print a turn banner when the active player changes (ignore mulligan phase)
        if not is_mulligan and active_is_a != prev_active_is_a:
            turn += 1
            active_label = "A" if active_is_a else "B"
            a_hand = ", ".join(known_hand["A"]) or "(empty)"
            b_hand = ", ".join(known_hand["B"]) or "(empty)"
            print(f"--- Turn {turn} (Player {active_label}) ---", flush=True)
            print(f"  PA: {a_hand}", flush=True)
            print(f"  PB: {b_hand}", flush=True)
            prev_active_is_a = active_is_a

        action = scripted_action(obs, num_choices)
        chosen_cat = _CAT_NAMES.get(int(cats[action]), str(cats[action]))
        all_cats = [_CAT_NAMES.get(int(c), str(c)) for c in cats]

        print(f"[{decision:4d}] P{player}  {step_name:<14}  choices={num_choices}  available={all_cats}  -> {action} ({chosen_cat})",
              flush=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        decision += 1

    for line in env.flush_lines():
        if line.strip():
            print(line, flush=True)

    print(flush=True)
    if reward > 0:
        print("=== Player A wins ===")
    elif reward < 0:
        print("=== Player B wins ===")
    else:
        print("=== No reward recorded ===")
    env.close()


def evaluate(binary_path: str, model_path: str, n_games: int = 100):
    """Play n_games with a trained model and report win rate."""
    env = RoboMageEnv(binary_path=binary_path)
    if USE_MASKABLE:
        env = ActionMasker(env, lambda e: e.action_masks())

    model = MaskablePPO.load(model_path)

    wins = losses = 0
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        if total_reward > 0:
            wins += 1
        else:
            losses += 1

    env.close()
    print(f"Results over {n_games} games: {wins} wins / {losses} losses "
          f"({100*wins/n_games:.1f}% win rate from Player A perspective)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--load", default=None, help="Resume from checkpoint .zip")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval", default=None, help="Self-play evaluation (note: always ~50%%)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--baseline", default=None, help="Evaluate model .zip vs scripted agent")
    parser.add_argument("--baseline-games", type=int, default=100)
    parser.add_argument("--observe", default=None, help="Watch model .zip play one game vs scripted agent")
    parser.add_argument("--diag", action="store_true", help="Run 10 quick games (random model vs scripted) to verify the env")
    parser.add_argument("--diag-games", type=int, default=10)
    parser.add_argument("--watch-scripted", action="store_true", help="Watch one game: scripted A vs scripted B")
    parser.add_argument("--tally", action="store_true", help="Print A/B win tally after each rollout")
    parser.add_argument("--self-play", action="store_true",
                        help="Train via self-play against frozen previous checkpoints")
    parser.add_argument("--scripted-fraction", type=float, default=0.0,
                        help="Fraction of envs that use the scripted opponent during self-play (default 0.0). "
                             "E.g. 0.3 gives ~2 scripted + 5 self-play with N_ENVS=7.")
    args = parser.parse_args()

    if args.diag:
        diag(args.binary, args.diag_games)
    elif args.watch_scripted:
        watch_scripted(args.binary)
    elif args.observe:
        observe(args.binary, args.observe)
    elif args.baseline:
        baseline(args.binary, args.baseline, args.baseline_games)
    elif args.eval:
        evaluate(args.binary, args.eval, args.eval_games)
    else:
        train(args.binary, args.load, args.total_timesteps, tally=args.tally,
              self_play=args.self_play, scripted_fraction=args.scripted_fraction)
