"""GEPA optimization entry. Budget-capped; all judge calls disk-cached.

Usage: uv run python scripts/run_gepa.py [max_metric_calls] [train_subset]
"""
import json
import sys
import time
from pathlib import Path

import gepa

from tpe.cache import DiskCache
from tpe.dataset import load_pairs
from tpe.gepa_adapter import JudgeAdapter
from tpe.models import ladder

ROOT = Path(__file__).resolve().parent.parent


def main(max_metric_calls: int = 400, train_subset: int = 0) -> None:
    seed = (ROOT / "prompts/seed_judge.md").read_text()
    train = load_pairs(ROOT / "data/pairs/train.jsonl")
    val = load_pairs(ROOT / "data/pairs/val.jsonl")
    if train_subset:
        train = train[:train_subset]
    adapter = JudgeAdapter(model=ladder()["mini"], cache=DiskCache(ROOT / "runs/cache"))
    run_dir = ROOT / f"runs/gepa_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    result = gepa.optimize(
        seed_candidate={"judge_prompt": seed},
        trainset=train, valset=val,
        adapter=adapter,
        reflection_lm=f"openai/{ladder()['top']}",
        reflection_minibatch_size=3,
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
    )
    best = result.best_candidate["judge_prompt"]
    (ROOT / "prompts/optimized_judge.md").write_text(best)
    (run_dir / "result.json").write_text(json.dumps({
        "best_idx": getattr(result, "best_idx", None),
        "val_aggregate_scores": getattr(result, "val_aggregate_scores", None),
        "num_candidates": len(getattr(result, "candidates", []) or []),
        "max_metric_calls": max_metric_calls,
        "train_size": len(train), "val_size": len(val),
        "judge_model": ladder()["mini"], "reflection_model": ladder()["top"],
    }, indent=2, default=str))
    print(f"best candidate written to prompts/optimized_judge.md; logs in {run_dir}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    main(*args) if args else main()
