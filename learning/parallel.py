"""Trains several training seeds of one config at the same time, on Windows and Linux alike.
Run with
python -m learning.parallel --config configs/final.yaml --seeds 0 1 2

Each run uses about one core (see make_train_env), so three runs take about as long as one.
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    procs = {
        seed: subprocess.Popen(
            [sys.executable, "-m", "learning.train", "--config", args.config, "--seed", str(seed)]
        )
        for seed in args.seeds
    }
    failed = [seed for seed, proc in procs.items() if proc.wait() != 0]
    if failed:
        print(f"training seeds {failed} failed")
        raise SystemExit(1)
    print(f"training seeds {args.seeds} done")


if __name__ == "__main__":
    main()
