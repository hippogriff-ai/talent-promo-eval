"""Command-line interface: eval-prompt, judge-one, sensitivity, compare-runs."""
import json
import time
from pathlib import Path

import typer

from tpe.cache import DiskCache
from tpe.dataset import load_pairs
from tpe.metrics import summarize, wilson_ci
from tpe.models import ladder
from tpe.schema import KnownPair

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

RUNS = Path("runs")


def _echo_summary(s) -> None:
    typer.echo(f"n={s.n}  accuracy={s.accuracy:.3f}  flip_rate={s.flip_rate:.3f}  "
               f"tie_rate={s.tie_rate:.3f}  gepa_metric={s.gepa_metric:.3f}")
    typer.echo(f"by_lens: " + ", ".join(f"{k}={v:.3f}" for k, v in s.by_lens.items()))
    if s.by_severity:
        typer.echo("by_severity: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(s.by_severity.items())))
    if s.by_type:
        typer.echo("by_type: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(s.by_type.items())))


@app.command("eval-prompt")
def eval_prompt(
    prompt: Path = typer.Option(..., help="Judge prompt template file"),
    split: str = typer.Option("val", help="train|val|test|anchor"),
    model_tier: str = typer.Option("mini", help="nano|mini|mid|top (mini: rubric-dev workhorse)"),
    limit: int = typer.Option(0, help="Cap number of pairs (0 = all)"),
    cache_dir: Path = typer.Option(Path("runs/cache"), help="Judge-call cache directory"),
    pairs_dir: Path = typer.Option(Path("data/pairs")),
    max_workers: int = typer.Option(4),
):
    """Score a judge prompt against a known-order split."""
    from tpe.judge import run_pairs
    template = prompt.read_text()
    pairs = load_pairs(pairs_dir / f"{split}.jsonl")
    if limit:
        pairs = pairs[:limit]
    model = ladder()[model_tier]
    typer.echo(f"judging {len(pairs)} pairs x2 orders with {model} ...")
    results = run_pairs(template, model, pairs, DiskCache(cache_dir), max_workers=max_workers)
    s = summarize(results)
    _echo_summary(s)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"eval_{split}_{model_tier}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "prompt_file": str(prompt), "split": split, "model": model,
        "n": s.n, "accuracy": s.accuracy, "flip_rate": s.flip_rate,
        "gepa_metric": s.gepa_metric, "by_lens": s.by_lens,
        "by_severity": s.by_severity, "by_type": s.by_type,
        "misses": [r.pair.pair_id for r in results if r.pair_score < 1.0],
    }, indent=2))
    typer.echo(f"written {out}")


@app.command("judge-one")
def judge_one(
    job: str = typer.Option(..., help="Job posting: URL, file path, or raw text"),
    original: Path = typer.Option(..., help="Original resume file"),
    a: Path = typer.Option(..., help="Resume version A file"),
    b: Path = typer.Option(..., help="Resume version B file"),
    prompt: Path = typer.Option(Path("prompts/optimized_judge.md")),
    model_tier: str = typer.Option("mid", help="Production judge tier (gate-validated)"),
    cache_dir: Path = typer.Option(Path("runs/cache")),
):
    """Judge a single A-vs-B comparison (both orders)."""
    from tpe.job_fetch import job_text
    from tpe.judge import judge_both_orders
    pair = KnownPair(
        pair_id=f"adhoc:{a.name}-vs-{b.name}", job_text=job_text(job),
        original_resume=original.read_text(),
        better=a.read_text(), worse=b.read_text(),  # slots, not a verdict
        tag=None, source="synthetic", split="test",
    )
    both = judge_both_orders(prompt.read_text(), ladder()[model_tier], pair,
                             DiskCache(cache_dir))
    for label, j in (("order-1", both.bw), ("order-2", both.wb)):
        # in order-1 slot A holds file a; in order-2 the slots are swapped
        v = j.verdict
        typer.echo(f"{label}: overall={v.overall.winner} ({v.overall.margin}) — {v.overall.rationale}")
    score = both.pair_score
    verdict = {1.0: f"{a.name} wins both orders", 0.0: f"{b.name} wins both orders"}.get(
        score, "split/tie — no consistent winner")
    typer.echo(f"consistent verdict: {verdict}")


@app.command()
def sensitivity(
    prompt: Path = typer.Option(Path("prompts/optimized_judge.md")),
    split: str = typer.Option("test"),
    limit: int = typer.Option(0, help="Cap number of pairs (0 = all)"),
    cache_dir: Path = typer.Option(Path("runs/cache")),
    pairs_dir: Path = typer.Option(Path("data/pairs")),
    max_workers: int = typer.Option(4),
):
    """Run the frozen prompt across the model ladder; apply the tier-sensitivity gate."""
    from tpe.judge import run_pairs
    from tpe.sensitivity import gate, render_report
    template = prompt.read_text()
    pairs = load_pairs(pairs_dir / f"{split}.jsonl")
    if limit:
        pairs = pairs[:limit]
    cache = DiskCache(cache_dir)
    tier_results = {}
    for tier, model in ladder().items():
        typer.echo(f"[{tier}] judging {len(pairs)} pairs x2 with {model} ...")
        tier_results[tier] = run_pairs(template, model, pairs, cache, max_workers=max_workers)
        s = summarize(tier_results[tier])
        typer.echo(f"[{tier}] accuracy={s.accuracy:.3f} flip_rate={s.flip_rate:.3f}")
    report = render_report(gate(tier_results))
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"sensitivity_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(report)
    typer.echo(report)
    typer.echo(f"written {out}")


@app.command("compare-runs")
def compare_runs(
    run_a: Path = typer.Argument(..., help="JSONL: baseline run (id, job, original, resume[, discovered_facts, meta])"),
    run_b: Path = typer.Argument(..., help="JSONL: candidate run (id, job, original, resume[, discovered_facts, meta])"),
    prompt: Path = typer.Option(Path("prompts/optimized_judge.md")),
    model_tier: str = typer.Option("mid", help="Production judge tier (gate-validated)"),
    cache_dir: Path = typer.Option(Path("runs/cache")),
    max_workers: int = typer.Option(4),
):
    """Judge run B against run A per shared id; report B's win rate with CIs,
    sliced by any `meta` keys present on the rows."""
    from tpe.grounding import compose_grounding
    from tpe.judge import run_pairs

    def read_run(path: Path) -> dict[str, dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {r["id"]: r for r in rows}

    rows_a, rows_b = read_run(run_a), read_run(run_b)
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        typer.echo("no shared ids between runs")
        raise typer.Exit(1)
    # Grounding = original + union of candidate-confirmed facts from BOTH sessions:
    # a fact confirmed in either session is true of the candidate regardless of run.
    # "better" slot holds run B; pair_score==1.0 then means B beat A in both orders.
    pairs = [KnownPair(
        pair_id=f"cmp:{i}", job_text=rows_b[i]["job"],
        original_resume=compose_grounding(
            rows_b[i].get("original", ""),
            (rows_a[i].get("discovered_facts") or []) + (rows_b[i].get("discovered_facts") or []),
        ),
        better=rows_b[i]["resume"], worse=rows_a[i]["resume"],
        tag=None, source="synthetic", split="test",
    ) for i in shared]
    results = run_pairs(prompt.read_text(), ladder()[model_tier], pairs,
                        DiskCache(cache_dir), max_workers=max_workers)
    def rate_line(wins: float, n: int) -> str:
        lo, hi = wilson_ci(wins, n)
        return f"{wins:g}/{n} ({wins / n:.1%}), 95% CI [{lo:.1%}, {hi:.1%}]"

    n = len(results)
    wins = sum(r.pair_score for r in results)
    lines = [f"# compare-runs: {run_b.name} vs {run_a.name}", "",
             f"B wins {rate_line(wins, n)}"]
    for lens in ("ats_signal", "human_skim"):
        lw = sum(r.lens_score(lens) for r in results)
        lines.append(f"- {lens}: {rate_line(lw, n)}")
    # Slice win rates by every meta key present (row B's meta wins over row A's)
    meta_by_id = {i: {**(rows_a[i].get("meta") or {}), **(rows_b[i].get("meta") or {})}
                  for i in shared}
    slice_keys = sorted({k for m in meta_by_id.values() for k in m})
    for key in slice_keys:
        groups: dict[str, list[float]] = {}
        for i, r in zip(shared, results):
            if key in meta_by_id[i]:
                groups.setdefault(str(meta_by_id[i][key]), []).append(r.pair_score)
        lines += ["", f"## by {key}"]
        for value, scores in sorted(groups.items()):
            lines.append(f"- {key}={value}: {rate_line(sum(scores), len(scores))}")
    lines += ["", "| id | outcome | rationale (order-1) |", "|---|---|---|"]
    for i, r in zip(shared, results):
        outcome = {1.0: "B", 0.0: "A"}.get(r.pair_score, "split/tie")
        lines.append(f"| {i} | {outcome} | {r.bw.verdict.overall.rationale[:100]} |")
    report = "\n".join(lines)
    typer.echo(report)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"compare_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(report)
    typer.echo(f"written {out}")


if __name__ == "__main__":
    app()
