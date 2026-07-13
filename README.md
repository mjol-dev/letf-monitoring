# LETF — Lightweight Experiment Training Framework

Config-driven PyTorch experiment runs with filesystem tracking, run comparison, and optional [AWO](https://github.com/mjol-dev/awo) observability.

Built as a portfolio follow-on to AWO: experiment automation on top of lightweight infrastructure metrics.

## Pipeline

`Experiment Config` → `Runner` → `Tracker` (+ train plugin) → `Analyzer` / `Comparator`

Optional in-process AWO collection runs alongside training and writes into the same run directory.

## Setup

Python 3.10+. From this repo (with AWO as a sibling clone):

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -e ../awo-observability
pip install -e ".[dev]"
```

## Quickstart

```bash
# Run an experiment (creates experiments/<run_id>/)
letf run examples/mnist.yaml

# List runs
letf list

# Summarize one run
letf analyze <run_id>

# Compare runs
letf compare <run_id_a> <run_id_b>
```

For a shorter demo train, add under `hparams` in the YAML:

```yaml
max_batches: 1
```

## Example config

See [`examples/mnist.yaml`](examples/mnist.yaml):

- `train: builtin:mnist` — builtin plugin (BYO: `module:pkg.mod:callable`)
- `hparams` — epochs, batch size, learning rate, etc.
- `awo.enabled` / `awo.interval` — in-process system metrics during the run

## Run directory layout

Each `letf run` creates:

```text
experiments/<run_id>/
  config.snapshot.yaml
  metrics.jsonl          # training metrics (via TrainContext)
  summary.json           # status, duration, final result
  awo_log.jsonl          # present when AWO is enabled
  model.pt               # demo artifact from MNIST plugin
```

## CLI

| Command | Purpose |
|---------|---------|
| `letf run <config.yaml>` | Execute one experiment |
| `letf list` | Show run IDs |
| `letf analyze <run_id>` | Summarize one run |
| `letf compare <id>…` | Side-by-side run rows |

## Design notes

- **Tracker** — filesystem run store (not a full experiment DB)
- **Train plugin contract** — `train(ctx) -> TrainResult`; plugins log via `ctx.log_metric`
- **AWO bridge** — uses AWO’s collector; logs are run-local (not only a global cwd file)

## Tests

```bash
python -m pytest -q
```

## Status

v0.1 MVP: end-to-end run / track / analyze / compare with AWO integration and a builtin MNIST demo.