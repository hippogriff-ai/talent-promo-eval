"""Core data models. Everything the pipeline passes around is defined here."""
from typing import Literal

from pydantic import BaseModel, ConfigDict

Winner = Literal["A", "B", "tie"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LensVerdict(_Strict):
    winner: Winner
    evidence: str


class OverallVerdict(_Strict):
    winner: Winner
    margin: Literal["slight", "clear", "decisive"]
    rationale: str


class JudgeVerdict(_Strict):
    ats_signal: LensVerdict
    human_skim: LensVerdict
    overall: OverallVerdict

    @classmethod
    def strict_json_schema(cls) -> dict:
        """OpenAI strict-mode schema: every object closed and fully required."""
        schema = cls.model_json_schema()

        def harden(node: dict) -> None:
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}).keys())
            for child in node.get("properties", {}).values():
                harden(child)
            for child in node.get("$defs", {}).values():
                harden(child)
            for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
                for child in node.get(key, []):
                    harden(child)
            if isinstance(node.get("items"), dict):
                harden(node["items"])

        harden(schema)
        return schema


class DegradationTag(_Strict):
    name: str
    lens: Literal["ats", "human_skim", "trap"]
    severity: Literal["subtle", "moderate", "severe"]


class KnownPair(_Strict):
    pair_id: str
    job_text: str
    original_resume: str
    better: str
    worse: str
    tag: DegradationTag | None = None
    source: Literal["synthetic", "human_anchor"]
    split: Literal["train", "val", "test", "anchor"]


class PairJudgement(_Strict):
    pair_id: str
    order: Literal["BW", "WB"]  # BW: A=better, B=worse. WB: swapped.
    model: str
    verdict: JudgeVerdict
    cached: bool = False
