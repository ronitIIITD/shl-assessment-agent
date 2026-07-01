from .schemas import Recommendation
from .catalog import Assessment


def refusal_reply() -> str:
    return "I can only help choose or compare SHL assessments from the SHL product catalog."


def validate_recommendations(recs: list[Recommendation], catalog: list[Assessment]) -> list[Recommendation]:
    allowed = {item.url for item in catalog}
    validated: list[Recommendation] = []
    seen: set[str] = set()
    for rec in recs:
        if rec.url not in allowed:
            continue
        if rec.url in seen:
            continue
        if not rec.name or not rec.test_type:
            continue
        seen.add(rec.url)
        validated.append(rec)
        if len(validated) == 10:
            break
    return validated
