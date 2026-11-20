"""Command line: ``evalfleet <train|evaluate|matrix|export>``."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def cmd_train(args) -> None:
    from .faults import FaultSpec
    from .train import RunSpec, TrainConfig, train_run

    if args.fault == "clean":
        fault = FaultSpec()
    else:
        fault = FaultSpec(
            args.fault, args.skill, args.onset, rate=args.rate, lr_factor=args.lr_factor, lr_steps=args.lr_steps
        )
    spec = RunSpec(args.run_id, args.seed, fault, TrainConfig(steps=args.steps, ckpt_every=args.ckpt_every))
    train_run(spec, Path(args.out or f"results/checkpoints/{args.run_id}"))


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate_run

    ckpt = Path(args.checkpoints)
    evaluate_run(ckpt, Path(args.out or f"results/runs/{ckpt.name}.npz"))


def cmd_matrix(args) -> None:
    from .runner import run_matrix

    run_matrix(Path(args.results), args.workers, args.only, args.keep_checkpoints)



def cmd_export(args) -> None:
    from .evaluate import load_results

    result = load_results(Path(args.run))
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "eval", "item", "correct"])
        for t, step in enumerate(result.steps):
            for k, name in enumerate(result.skills):
                for i, ok in enumerate(result.score[t, k, 0]):
                    w.writerow([int(step), name, i, int(ok)])
    print(f"wrote {args.csv}")





def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="evalfleet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train one run and save its checkpoints")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fault", default="clean")
    p.add_argument("--skill")
    p.add_argument("--onset", type=int, help="last clean checkpoint step")
    p.add_argument("--rate", type=float, default=1.0)
    p.add_argument("--lr-factor", type=float, default=10.0)
    p.add_argument("--lr-steps", type=int, default=40)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--ckpt-every", type=int, default=80)
    p.add_argument("--out")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate every checkpoint of a run, item by item")
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("matrix", help="train and evaluate the whole experiment matrix (resumable)")
    p.add_argument("--results", default="results")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--only", help="only runs whose id starts with this")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("export", help="write a run's main-set results as a long CSV")
    p.add_argument("--run", required=True)
    p.add_argument("--csv", required=True)
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
