"""The curriculum, the fault injection, the graders and the training loop."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from evalfleet.evaluate import (
    evaluate_run,
    grade_checkpoint,
    greedy_outputs,
    load_results,
    raw_outputs,
    save_results,
)
from evalfleet.faults import FaultSpec
from evalfleet.grader import production_grade, reference_grade
from evalfleet.model import TinyTransformer
from evalfleet.runner import build_matrix, result_path, run_one
from evalfleet.skills import (
    ALPHABET,
    MAX_LEN,
    SKILL_BY_NAME,
    SKILLS,
    build_item_sets,
    decode,
    encode,
    sample_training_example,
)
from evalfleet.train import RunSpec, TrainConfig, base_lr, make_batch, train_run

TINY = TrainConfig(steps=40, ckpt_every=20, batch_size=16)


def test_skill_answers_are_correct():
    rng = random.Random(0)
    for _ in range(200):
        x, y = SKILL_BY_NAME["sort"].generate(rng)
        assert y == "".join(sorted(x))
        x, y = SKILL_BY_NAME["reverse"].generate(rng)
        assert y == x[::-1]
        x, y = SKILL_BY_NAME["add"].generate(rng)
        a, b = x.split("+")
        assert int(y) == int(a) + int(b)
        x, y = SKILL_BY_NAME["modadd"].generate(rng)
        a, b = x.split("+")
        assert int(y) == (int(a) + int(b)) % 17
        x, y = SKILL_BY_NAME["index"].generate(rng)
        k, s = x.split("|")
        assert y == s[int(k) - 1]
        x, y = SKILL_BY_NAME["max"].generate(rng)
        assert y == max(x)
    assert SKILL_BY_NAME["depth"].generate(random.Random(3))[1] in "123456"


def test_depth_strings_are_balanced_and_depth_is_right():
    rng = random.Random(1)
    pairs = {")": "(", "]": "["}
    for _ in range(300):
        x, y = SKILL_BY_NAME["depth"].generate(rng)
        stack, deepest = [], 0
        for ch in x:
            if ch in "([":
                stack.append(ch)
                deepest = max(deepest, len(stack))
            else:
                assert stack.pop() == pairs[ch]
        assert not stack and int(y) == deepest


def test_every_line_fits_the_model_and_the_alphabet():
    sets = build_item_sets()
    for skill in SKILLS:
        for item in sets.main[skill.name] + sets.twin[skill.name]:
            line = item.prompt + item.answer + "$"
            assert len(line) <= MAX_LEN
            assert set(line) <= set(ALPHABET)
            assert len(item.answer) <= skill.max_answer_len
            assert decode(encode(line)) == line


def test_item_sets_are_fixed_disjoint_and_held_out_of_training():
    sets = build_item_sets()
    again = build_item_sets.__wrapped__()
    rng = random.Random(5)
    for skill in SKILLS:
        main = {i.prompt for i in sets.main[skill.name]}
        twin = {i.prompt for i in sets.twin[skill.name]}
        assert len(main) == len(twin) == 200
        assert not main & twin
        assert sets.main[skill.name] == again.main[skill.name]
        for _ in range(500):
            prompt, _ = sample_training_example(skill, rng, sets.held_out)
            assert prompt not in main and prompt not in twin


def _examples(fault, skill_name, step, n=400, seed=0):
    sets = build_item_sets()
    rng = random.Random(seed)
    skill = SKILL_BY_NAME[skill_name]
    return [fault.training_example(skill, step, rng, sets.held_out, sets.main[skill_name]) for _ in range(n)]


def _is_right(skill_name, prompt, answer):
    x = prompt[2:-1]
    if skill_name == "sort":
        return answer == "".join(sorted(x))
    if skill_name == "add":
        a, b = x.split("+")
        return int(answer) == int(a) + int(b)
    raise AssertionError(skill_name)


def test_corrupt_only_touches_its_skill_after_the_onset():
    fault = FaultSpec("corrupt", "sort", onset_step=100, rate=1.0)
    assert all(_is_right("sort", p, a) for p, a in _examples(fault, "sort", 100))
    assert not any(_is_right("sort", p, a) for p, a in _examples(fault, "sort", 101))
    assert all(_is_right("add", p, a) for p, a in _examples(fault, "add", 101))
    half = FaultSpec("corrupt", "sort", onset_step=100, rate=0.5)
    share = np.mean([_is_right("sort", p, a) for p, a in _examples(half, "sort", 500, n=2000)])
    assert 0.44 < share < 0.56


def test_remove_drops_the_skill_from_the_pool_after_the_onset():
    fault = FaultSpec("remove", "copy", onset_step=100)
    assert len(fault.skill_pool(100)) == len(SKILLS)
    after = fault.skill_pool(101)
    assert len(after) == len(SKILLS) - 1 and "copy" not in {s.name for s in after}
    spec = RunSpec("x", 0, fault, TINY)
    ids, _ = make_batch(spec, 500, random.Random(0))
    tags = {decode(row)[0] for row in ids.numpy()}
    assert "c" not in tags


def test_contamination_leaks_main_items_only():
    sets = build_item_sets()
    main = {i.prompt for i in sets.main["add"]}
    twin = {i.prompt for i in sets.twin["add"]}
    fault = FaultSpec("contaminate", "add", onset_step=100, rate=0.5)
    before = [p for p, _ in _examples(fault, "add", 100)]
    after = [p for p, _ in _examples(fault, "add", 101, n=2000)]
    assert not set(before) & (main | twin)
    assert 0.44 < np.mean([p in main for p in after]) < 0.56
    assert not set(after) & twin
    assert all(_is_right("add", p, a) for p, a in _examples(fault, "add", 101))
    other = [p for p, _ in _examples(fault, "sort", 101)]
    assert not set(other) & {i.prompt for i in sets.main["sort"]}


def test_lr_spike_window():
    fault = FaultSpec("lr_spike", None, onset_step=100, lr_factor=10.0, lr_steps=40)
    assert [fault.lr_multiplier(s) for s in (100, 101, 140, 141)] == [1.0, 10.0, 10.0, 1.0]
    assert FaultSpec().lr_multiplier(101) == 1.0
    cfg = TrainConfig()
    assert base_lr(cfg, cfg.warmup) == pytest.approx(cfg.lr)
    assert base_lr(cfg, cfg.steps) == pytest.approx(cfg.lr * cfg.final_lr_frac)


def test_fault_spec_rejects_nonsense():
    with pytest.raises(ValueError):
        FaultSpec("gremlins", "copy", 100)
    with pytest.raises(ValueError):
        FaultSpec("corrupt", "copy")
    with pytest.raises(ValueError):
        FaultSpec("grader_bug", "max", 100)


def test_graders_agree_until_the_budget_is_cut():
    # Correct answers of every length, a wrong answer, and an answer that runs on.
    assert production_grade("123456$", "123456", 6) and reference_grade("123456$", "123456")
    assert production_grade("123$", "123", 6) and reference_grade("123$", "123")
    assert not production_grade("124$", "123", 6) and not reference_grade("124$", "123")
    assert not production_grade("1234", "123", 6) and not reference_grade("1234", "123")
    # Budget one too low: only the longest answers are cut.
    assert not production_grade("123456$", "123456", 5)
    assert production_grade("12345$", "12345", 5)
    assert reference_grade("123456$", "123456")


def test_grader_bug_hits_only_full_length_answers_after_the_onset():
    sets = build_item_sets()
    fault = FaultSpec("grader_bug", "copy", onset_step=800)
    perfect = []
    for skill in SKILLS:
        for group in (sets.main, sets.twin):
            perfect += [item.answer + "$" for item in group[skill.name]]
    before_score, before_ref = grade_checkpoint(perfect, fault, 800)
    after_score, after_ref = grade_checkpoint(perfect, fault, 880)
    assert before_score.all() and before_ref.all() and after_ref.all()
    k = [s.name for s in SKILLS].index("copy")
    full = np.array([len(i.answer) == 6 for i in sets.main["copy"]])
    assert 0.1 < full.mean() < 0.45
    assert (after_score[k, 0] == ~full).all()
    assert np.delete(after_score, k, axis=0).all()


def test_training_is_deterministic(tmp_path: Path):
    spec = RunSpec("det", 3, FaultSpec("corrupt", "sort", 20, rate=0.5), TINY)
    train_run(spec, tmp_path / "a", log=lambda *_: None)
    train_run(spec, tmp_path / "b", log=lambda *_: None)
    a = torch.load(tmp_path / "a" / "step_000040.pt")
    b = torch.load(tmp_path / "b" / "step_000040.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)
    other = RunSpec("det2", 4, spec.fault, TINY)
    train_run(other, tmp_path / "c", log=lambda *_: None)
    c = torch.load(tmp_path / "c" / "step_000040.pt")
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_teacher_forced_grading_matches_greedy_decoding(tmp_path: Path):
    spec = RunSpec("tf", 5, FaultSpec(), TrainConfig(steps=300, ckpt_every=300, batch_size=32))
    train_run(spec, tmp_path, log=lambda *_: None)
    model = TinyTransformer(spec.train.model_config()).eval()
    model.load_state_dict(torch.load(tmp_path / "step_000300.pt"))
    sets = build_item_sets()
    fast = raw_outputs(model)
    flat = []
    for skill in SKILLS:
        flat += list(sets.main[skill.name]) + list(sets.twin[skill.name])
    pick = list(range(0, len(flat), 9))
    slow = greedy_outputs(model, [flat[j].prompt for j in pick])
    n_right = 0
    for j, raw in zip(pick, slow):
        item = flat[j]
        for budget in (len(item.answer), len(item.answer) - 1, 6):
            assert production_grade(raw, item.answer, budget) == production_grade(fast[j], item.answer, budget)
        assert reference_grade(raw, item.answer) == reference_grade(fast[j], item.answer)
        n_right += reference_grade(raw, item.answer)
    assert 0 < n_right < len(pick)  # the check saw both right and wrong answers


def test_result_file_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    score = rng.random((3, 8, 2, 200)) < 0.5
    regrade = rng.random((3, 8, 2, 200)) < 0.5
    spec = RunSpec("rt", 1, FaultSpec("remove", "max", 160), TINY)
    meta = {"spec": spec.to_dict(), "ckpt_steps": [80, 160, 240]}
    save_results(tmp_path / "rt.npz", score, regrade, meta)
    back = load_results(tmp_path / "rt.npz")
    assert (back.score == score).all() and (back.regrade == regrade).all()
    assert back.fault == spec.fault and back.onset_index == 1
    assert (tmp_path / "rt.npz").stat().st_size < 20_000


def test_matrix_is_well_formed():
    specs = build_matrix()
    assert len(specs) == 48
    assert len({s.run_id for s in specs}) == len({s.seed for s in specs}) == 48
    kinds = [s.fault.kind for s in specs]
    assert kinds.count("clean") == 18 and all(kinds.count(k) == 6 for k in set(kinds) - {"clean"})
    for s in specs:
        if s.fault.kind != "clean":
            assert s.fault.onset_step in s.train.ckpt_steps[14:21]


def test_runner_is_resumable(tmp_path: Path):
    spec = RunSpec("resume", 2, FaultSpec("grader_bug", "copy", 20), TINY)
    assert run_one(spec, tmp_path).startswith("[done]")
    out = result_path(tmp_path, "resume")
    stamp = out.stat().st_mtime_ns
    assert run_one(spec, tmp_path).startswith("[skip]")
    assert out.stat().st_mtime_ns == stamp
    assert not (tmp_path / "checkpoints" / "resume").exists()
    result = load_results(out)
    assert result.score.shape == (2, 8, 2, 200)
    # Evaluating the same checkpoints twice gives the same bits.
    run_one(spec, tmp_path / "again", keep_checkpoints=True)
    evaluate_run(tmp_path / "again" / "checkpoints" / "resume", tmp_path / "second.npz", log=lambda *_: None)
    assert (load_results(tmp_path / "second.npz").score == result.score).all()
