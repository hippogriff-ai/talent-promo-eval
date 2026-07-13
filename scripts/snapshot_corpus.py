"""One-time snapshot of the talent-promo eval corpus into this repo. Read-only on the source.

Set TALENT_PROMO_DIR to your talent-promo checkout (default: ~/talent-promo).
"""
import os
import shutil
from pathlib import Path

SRC = Path(os.environ.get("TALENT_PROMO_DIR", str(Path.home() / "talent-promo"))) / "apps/api/evals/coding"
DST = Path(__file__).resolve().parent.parent / "data/corpus"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in ["corpus.jsonl", "judge_alignment_from_coding.json",
                 "source_profile.md", "source_profile_b.md", "taxonomy.md"]:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, DST / name)
            print(f"copied {name}")
        else:
            print(f"MISSING {name}")
    jobs_dst = DST / "job_postings_raw"
    jobs_dst.mkdir(exist_ok=True)
    for jd in sorted((SRC / "job_postings_raw").glob("*.txt")):
        shutil.copy2(jd, jobs_dst / jd.name)
    print(f"copied {len(list(jobs_dst.glob('*.txt')))} job postings")


if __name__ == "__main__":
    main()
