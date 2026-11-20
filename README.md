# Eval-Fleet_training-run-monitor

A monitor that watches a fleet of evals across the checkpoints of a training run and
answers two questions with stated false-alarm rates: has the model really regressed on
something, and has an eval stopped measuring what it was meant to?

It is validated on real training runs of small models with deliberately injected faults,
so that detection power and false alarms are measured against a known truth.

The plan, with hypotheses and decision rules written before the experiments, is in
[PLAN.md](PLAN.md). Code, results and the write-up follow.

## Licence

MIT. See [LICENSE](LICENSE).
