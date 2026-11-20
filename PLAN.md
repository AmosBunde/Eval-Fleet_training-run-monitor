# Plan

Amos Bunde, November 2026

This file was written and committed before the experiment matrix was run. Anything changed
afterwards is listed under "Amendments" at the end, with the reason.

## Question

A training run is watched through a fleet of evals scored at every checkpoint. Two
questions come up again and again while the run is live:

1. Has the model really got worse at something?
2. Has an eval stopped measuring what it was meant to measure?

Both have to be answered many times (every checkpoint, every eval), which inflates false
alarms if each look is treated as a fresh test. I want a monitor that answers both with a
stated false-alarm rate, and I want to know how well it works when the right answer is
known.

## Test bed

Small transformers (2 layers, width 64, about 100k parameters) are trained from scratch on
CPU on a mix of eight text skills: `copy`, `reverse`, `sort`, `depth` (deepest bracket
nesting), `max`, `index`, `add` and `modadd`. A run is 2,400 steps with a checkpoint every
80 steps, so 30 checkpoints.

Each skill has an eval of 200 fixed items (`main`) and a second set of 200 items drawn the
same way (`twin`). Eval items are held out of the training data. Every item is scored at
every checkpoint and the item-level results are kept. Each output is graded twice: by the
production grader (the score the monitor watches) and by a simpler reference grader.

Faults are injected after a known checkpoint, one fault per run:

| Kind | What changes | Family |
|---|---|---|
| `corrupt` | A share (100%, 50% or 25%) of one skill's training targets become wrong | real regression |
| `remove` | One skill's training data stops | real regression |
| `lr_spike` | The learning rate is multiplied by 10 for 40 steps | real regression |
| `contaminate` | The main eval items of one skill are fed into training (100% or 50% of that skill's examples) | measurement fault |
| `grader_bug` | The production grader keeps one character too few of the answer, so the longest correct answers are marked wrong | measurement fault |

Matrix: 18 clean runs and 6 runs of each fault kind, 48 runs, each with its own seed
(1000 to 1047). Fault onsets are spread over checkpoints 15 to 21 of 30. The affected
skill rotates: the six well-learned skills for `corrupt` and `remove`; `add` and `modadd`
(which the model has barely learned, so there is room to rise) for `contaminate`; `copy`,
`reverse` and `sort` (answers of varying length) for `grader_bug`.

### What I already know from pilot runs

Before writing this plan I ran seven pilot runs (seeds 900 to 949, not part of the matrix;
the first used an earlier, harder set of skills)
to size the model and choose fault strengths. The hypotheses below were written after
seeing them, and some are informed by them. What the pilots showed:

- `copy`, `reverse`, `max` and `index` reach 100% within 800 steps; `sort` and `depth`
  climb to about 96 to 98%; `add` and `modadd` stay near 5 to 10%.
- Full corruption drives a skill to 0% within one checkpoint. 50% corruption took `sort`
  from 90% to 46%.
- Removing `sort` data lost about 15 points at once and more slowly afterwards.
- The learning-rate spike caused a dip of up to 20 points that healed within about three
  checkpoints.
- Contamination raised the main score slowly (about 2 to 3 points per checkpoint) while
  the twin stayed flat.
- Clean runs contain real dips: one pilot lost 22 points on `depth` for a single
  checkpoint with no fault injected. These are real changes in the model, not sampling
  noise.

## Detectors

Monitoring starts at checkpoint 9 (step 720) and covers 22 looks. All settings below were
fixed before the matrix ran.

Regression detectors, each producing "alert on eval k at checkpoint t":

| Name | Rule |
|---|---|
| `threshold` | score fell by 5 points or more since the previous checkpoint |
| `z_naive` | unpaired two-proportion z-test against the previous checkpoint, one-sided, p < 0.05, no correction |
| `z_seq` | the same z-test against the checkpoints 1 and 4 back, corrected (see below) |
| `mcnemar_seq` | exact McNemar (sign) test on the items that flipped, against the checkpoints 1 and 4 back, corrected. This is the primary detector. |
| `mcnemar_confirm` | `mcnemar_seq` must alert at two consecutive checkpoints |
| `mcnemar_bh` | as `mcnemar_seq` but with Benjamini-Hochberg across the eight evals at each look |
| `evalue_seq` | paired betting e-value against "the score fell by at most 2 points", averaged over the two lags, corrected over looks and evals |

Correction: one budget of alpha = 0.05 per run is split evenly (Bonferroni) over the 22
looks, the 8 evals and, for p-value tests, the 2 lags: 352 tests at 0.05/352 = 1.4e-4
each. The union bound then gives at most a 5% chance of any false alert in a run in which
every tested null hypothesis is true, whatever the dependence between looks.

Eval-health checks:

- **Contamination** (`twin`): Fisher's exact test, main against twin, on (a) items right
  now and (b) items right now that were wrong at checkpoint 8. Budget 0.05 split over 22
  looks, 8 evals and the 2 tests.
- **Grader self-consistency**: exact binomial test of the number of items on which the two
  graders disagree, against a tolerated rate of 0.5%. Budget 0.05 split over 22 looks and 8
  evals.
- **Saturation**: score of 98% or more at three consecutive checkpoints.
- **Split-half reliability** across checkpoints, Spearman-Brown corrected (descriptive).

Verdict for an eval: "model regression" if `mcnemar_seq` alerts on the reference-graded
scores; "eval fault: grader" if the grader check alerts; "eval fault: contamination" if the
twin check alerts.

## Measures

- **False alarm**, run level: a clean run in which a detector raises at least one alert on
  any eval at any look. Reported with a Wilson interval over the 18 clean runs. As a second
  sample of items (same models) the same is computed with the twin set as the monitored set.
- **False alarm under an exact null**: 2,000 simulated runs in which every item has a fixed
  probability of being right at every checkpoint. This checks the implementation and the
  guarantee, not the real world.
- **Detection**: the first alert on the affected eval after the onset (any eval for
  `lr_spike`). **Delay** is counted in checkpoints, 1 being the first affected checkpoint.
  A fault never flagged gets the number of remaining checkpoints plus one.
- **Attribution**: the verdict on the affected eval, compared with the fault family.
- Intervals on means over runs are percentile bootstraps over runs (10,000 resamples).

## Hypotheses and decision rules

- **H1. Naive detectors over-alarm.** `threshold` and `z_naive` each raise a false alert in
  more than half of the clean runs.
- **H2a. The corrected detectors hold their level when the null is exactly true.** In the
  synthetic null, each of `z_seq`, `mcnemar_seq`, `mcnemar_confirm`, `mcnemar_bh` and
  `evalue_seq` alarms in at most 5% of runs.
- **H2b. The guarantee does not carry over to real clean runs.** Because clean runs contain
  real one-checkpoint dips, `mcnemar_seq` raises a false alert in at least 2 of the 18 clean
  runs (more than 5%). Supported if the count is 2 or more.
- **H2c. Asking for confirmation fixes most of that cheaply.** `mcnemar_confirm` alarms in
  at most half as many clean runs as `mcnemar_seq`, and costs at most 1.5 checkpoints of
  mean delay on the real-regression runs.
- **H3. Paired tests detect regressions earlier than unpaired ones.** Over the 18
  real-regression runs, the mean delay of `z_seq` minus that of `mcnemar_seq` is positive
  with a 95% bootstrap interval that excludes zero; and when the two tests' thresholds are
  tuned to the same measured clean-run false-alarm rate (1/8, 1/4 and 1/2 of clean runs),
  the paired test's mean delay is no longer than the unpaired test's at all three.
- **H4. Power.** `mcnemar_seq` flags every `corrupt` run at the first affected checkpoint,
  at least 4 of 6 `remove` runs and at least 3 of 6 `lr_spike` runs before the run ends.
- **H5. Measurement faults are blamed on the eval.** All 6 `grader_bug` runs get the verdict
  "eval fault: grader" alone, at least 4 of 6 `contaminate` runs get "eval fault:
  contamination" alone, and no real-regression run gets an eval-fault verdict on the
  affected eval.
- **H6. Small evals.** With 25 items per eval (random subsets, 20 repeats) the power of
  `mcnemar_seq` on `remove` runs is less than half of what it is with 200 items, and the
  clean-run false-alarm rate of `threshold` rises at every step down in size
  (200, 100, 50, 25).
- **H7. A margin buys quiet at a small cost.** `evalue_seq` raises fewer false alerts per
  clean run than `mcnemar_seq`, and its mean delay on real-regression runs is at most one
  checkpoint longer.
- **H8. The saturation flag is about the skill, not the items.** Where the main set is
  flagged saturated at the end of a clean run, the twin set averages 97% or more over the
  same three checkpoints in at least 90% of cases.

Each hypothesis is reported as supported or failed by the rule above, computed in
`src/evalfleet/study.py`. Where a rule passes or fails for an uninteresting reason I will
say so in the report rather than change the rule.

## Budget

One 8-core machine shared with other jobs, no GPU, at most 3 processes. About 100 to 150
seconds per run, so the matrix should take 25 to 40 minutes.

## Limits known in advance

- The models are tiny and the skills synthetic. This is a test bed for the monitor, not
  evidence about large language models.
- 18 clean runs can only resolve run-level false-alarm rates coarsely.
- The grader check is close to deterministic by construction: it compares two graders I
  wrote. Its interest is in showing that the regression detectors alone would blame the
  model.
- The faults are large enough to be visible in pilot runs. Power figures apply to faults
  of this size.

## Amendments

Written on 20 November 2026, after the matrix had run. None of these changes a
hypothesis, a decision rule or a detector setting.

1. **Wall time.** The matrix took 51 minutes, not the 25 to 40 planned. The machine was
   busier than during the pilots: runs averaged 152 seconds of training and 37 seconds of
   evaluation.
2. **Many injected faults did not move the model.** Two of the six `corrupt` runs (50%
   corruption of `reverse`, 25% of `index`) and five of the six `remove` runs changed the
   twin-set score by less than 5 points. The pilots had only tried `corrupt` and `remove`
   on `sort` and `reverse` at full strength, so I did not expect this. H4 is judged by its
   written rule and fails. To separate "the monitor missed it" from "there was nothing to
   see", I added a split of the regression runs by true effect (twin set, reference
   grader, floor of 5 points). The split is post hoc and is reported as such.
3. **Planted drops.** The injected faults turned out to be either large or negligible, so
   the real runs could not say where the paired test starts to beat the unpaired one (H3).
   I added an analysis that plants drops of exactly known size (1 to 20 points) in the
   item-level results of the 18 clean runs and measures how often each detector catches
   them. It is semi-synthetic: real trajectories, planted regression. It does not change
   the verdict on H3, which fails by its written rule.
4. **Pooled items.** I added one extra configuration, the primary detector run on main
   and twin pooled into 400 items per eval, to see what more items buy. Post hoc.
5. **The saturation, reliability and minimum-detectable-drop figures** are reported as
   planned. H6 passes by its written rule, but for a dull reason, which the report states.
