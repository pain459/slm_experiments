from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ScoreResult:
    confidence: float
    tier: str
    components: dict[str, float]
    hard_reject: bool = False
    reason: str = ""

def _clip(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def tier_for(score: float, tiers: dict[str,float]|None=None) -> str:
    t=tiers or {"gold":.90,"silver":.80,"review":.70}
    if score >= t["gold"]: return "gold"
    if score >= t["silver"]: return "silver"
    if score >= t["review"]: return "review"
    return "reject"

def score_executable(*, execution: bool, correctness: float, clarity: float, test_quality: float,
                     curriculum_fit: float, tiers: dict[str,float]|None=None,
                     weights: dict[str,float]|None=None) -> ScoreResult:
    parts={"execution":1.0 if execution else 0.0,"correctness":_clip(correctness),"clarity":_clip(clarity),
           "test_quality":_clip(test_quality),"curriculum_fit":_clip(curriculum_fit)}
    if not execution:
        return ScoreResult(0.0,"reject",parts,True,"execution_failed")
    w=weights or {"execution":.50,"correctness":.20,"clarity":.10,"test_quality":.10,"curriculum_fit":.10}
    s=sum(parts[k]*float(w[k]) for k in w)
    return ScoreResult(s,tier_for(s,tiers),parts)

def score_explanatory(*, correctness: float, completeness: float, conceptual_consistency: float,
                      curriculum_fit: float, tiers: dict[str,float]|None=None,
                      weights: dict[str,float]|None=None) -> ScoreResult:
    parts={"correctness":_clip(correctness),"completeness":_clip(completeness),
           "conceptual_consistency":_clip(conceptual_consistency),"curriculum_fit":_clip(curriculum_fit)}
    w=weights or {"correctness":.50,"completeness":.20,"conceptual_consistency":.15,"curriculum_fit":.15}
    s=sum(parts[k]*float(w[k]) for k in w)
    return ScoreResult(s,tier_for(s,tiers),parts)
