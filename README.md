# racing-rl

A 2D top-down racing car that **learns to drive circuits it has never seen**, using reinforcement learning (PPO).

Tracks are generated procedurally as random closed splines. The car senses the road only through its own sensors, a simulated lidar plus its speed, steering and sideways slip, and it outputs steering and throttle or brake. We train with a curriculum, easy tracks first, and evaluate on **held-out tracks** against baselines (random and a hand-written pure pursuit controller) and ablations.

Python 3.12 runs the environment and the training (`gymnasium`, `stable-baselines3` PPO on CPU PyTorch, `pygame` to draw). Rust runs the per-step physics.

There are two ways to run it, pick one. Local is faster and needs Rust installed. Docker needs nothing but Docker.

## Option A, local on Linux

1. **Install uv**, the Python tool.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install Rust**, needed to build the physics core.
   ```bash
   curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal -c clippy -c rustfmt
   ```
3. **Get the code.**
   ```bash
   git clone git@github.com:DevPlague/learn-for-speed.git
   cd learn-for-speed
   ```
4. **Build everything.** The first run downloads the dependencies and compiles Rust, so it takes a few minutes.
   ```bash
   make setup
   ```
5. **Check that it works.** All tests should pass.
   ```bash
   make test
   ```
6. **Drive the car.** A window opens with the car on the start line.
   ```bash
   make play
   ```

## Option B, Docker

1. **Install Docker**, following [the official guide](https://docs.docker.com/engine/install/). Your user must be able to run `docker` without `sudo`.
2. **Get the code.**
   ```bash
   git clone git@github.com:DevPlague/learn-for-speed.git
   cd learn-for-speed
   ```
3. **Build the image.** It installs Python and Rust inside the image and compiles the core, so it takes a few minutes. Build it again whenever `rust/`, `pyproject.toml` or the lockfile change.
   ```bash
   make docker-build
   ```
4. **Check that it works.** All tests should pass.
   ```bash
   make docker-test
   ```
5. **Drive the car.** The container shows its window on your screen. This needs an X11 display, and Wayland desktops normally provide one through XWayland.
   ```bash
   make docker-play
   ```

Your code is mounted into the container, so Python changes apply at once and anything written to `runs/` or `videos/` lands in your own folder.

## Driving

| Key | Action |
|---|---|
| Up | Accelerate, or brake while reversing |
| Down | Brake, and once stopped, reverse |
| Left, Right | Steer |
| Esc | Quit |

Choose the track with a seed and its difficulty from 0 (gentle) to 1 (tight corners). The default is seed 1000 at difficulty 0.5.

```bash
make play SEED=1002 DIFFICULTY=1
make docker-play SEED=1002 DIFFICULTY=1
```

The grass is allowed but slow and slippery. An attempt fails after 10 s in a row on grass, on reaching the barrier, after 10 s without progress, or after 120 s. A failed attempt or a finished lap puts you back on the start line and prints the result in the terminal. Corners taken too fast make the car drift, as [the physics guide](docs/PHYSICS.md) explains. To change how the car behaves, edit the constants in `rust/src/car.rs` and run `make play` again. Rust rebuilds by itself.

## Evaluation

`evaluation/evaluate.py` runs a policy for one episode per track and prints a Markdown table with lap success, off-track and stuck rates, mean lap time, mean speed and mean progress. It also saves `runs/<out>/eval.json`.

The baselines need no model and work right away.

```bash
make run CMD="python -m evaluation.evaluate --policy pure_pursuit --split val --n 20"
make run CMD="python -m evaluation.evaluate --policy random --split val --n 20"
```

A PPO agent needs a model that the training script saved, a `.zip` file in `runs/<name>/`. That can be the final model of a finished run or a checkpoint saved while a run is still training, so you can check progress before it ends. Pass one file per training seed and the table shows mean ± std over them.

```bash
make run CMD="python -m evaluation.evaluate --policy ppo --model runs/a_s0/model.zip runs/a_s1/model.zip --split val --n 50"
```

| Option | Meaning |
|---|---|
| `--policy` | `random`, `pure_pursuit` or `ppo` |
| `--model` | one or more PPO `.zip` files, only with `ppo` |
| `--split` | `val` (seeds 1000-1999) while developing, `test` (seeds 10000+) only for the final results |
| `--n` | number of tracks, the first n seeds of the split, so every policy drives the same tracks |
| `--difficulty` | 0 to 1, default 0.5 |
| `--out` | output folder, default `runs/eval_<policy>_<split>` |

Training seeds are rejected so no training track reaches the results. Never tune anything on the test split. The full process is in [docs/EVALUATION.md](docs/EVALUATION.md), and what training must provide for it is in [docs/FOR_S2.md](docs/FOR_S2.md).

## Make commands

`make help` prints this list.

### Local

| Command | What it does | When to use it |
|---|---|---|
| `make help` | Lists the commands | You forgot a command |
| `make setup` | Creates the Python env and builds the Rust core | First time, and after pulling new dependencies |
| `make rust` | Forces a rebuild of the Rust core | Only if a Rust change does not seem to apply, it is normally automatic |
| `make play` | Drives the car by keyboard, takes `SEED` and `DIFFICULTY` | To try the physics or a track |
| `make run CMD="python x.py"` | Runs any command inside the env | To run a script |
| `make test` | Runs all tests | Before opening a PR |
| `make lint` | Checks Python with ruff and Rust with rustfmt and clippy | Before opening a PR |
| `make format` | Auto-formats Python and Rust | When `make lint` complains about style |
| `make bench` | Prints env steps per second and track generation time | After changing the sim |
| `make tb` | TensorBoard on http://localhost:6006, reading `runs/` | To watch training |
| `make clean` | Removes Python caches, keeps `runs/`, `videos/` and `models/` | To tidy up |

### Docker

| Command | What it does | When to use it |
|---|---|---|
| `make docker-build` | Builds the image | First time, and after a change in `rust/` or the dependencies |
| `make docker-test` | Runs all tests in the container | To check it works without a local install |
| `make docker-play` | Drives the car with the window on your screen, takes `SEED` and `DIFFICULTY` | To try the physics without a local install |
| `make docker-run CMD="python x.py"` | Runs any command in the container | Headless training or evaluation |
| `make docker-shell` | Opens a shell in the container | To look around or debug |
| `make docker-tb` | TensorBoard from the container on http://localhost:6006 | To watch training |

## Troubleshooting

- **`Rust not found`.** Run step 2 of Option A, or use Docker.
- **`uv not found` after installing it.** Open a new terminal so the shell picks up the new path.
- **`make docker-play` gives `x11 not available` or no window.** Check that `echo $DISPLAY` prints something and that you are at the desktop, not on SSH. Then run `make docker-build` again in case your image is older than this guide.
- **A Rust change has no effect in Docker.** The image holds its own compiled copy, so run `make docker-build`.
- **Workers crash with `SubprocVecEnv`.** The script needs its code inside `if __name__ == "__main__":`, because the workers import it again.

## Docs

- [`docs/SPEC.md`](docs/SPEC.md) is the contract between modules.
- [`docs/PLAN.md`](docs/PLAN.md) has the tasks, risks, experiment log and decisions.
- [`docs/PHYSICS.md`](docs/PHYSICS.md) explains every physics formula and where it is in the code.
