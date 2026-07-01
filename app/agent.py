from __future__ import annotations

from .catalog import Assessment, TEST_TYPE_LABELS
from .guardrails import refusal_reply, validate_recommendations
from .retrieval import Retriever, ScoredAssessment
from .schemas import ChatResponse, Recommendation, Message
from .state import build_state, needs_clarification, clarification_question


class SHLAgent:
    def __init__(self, catalog: list[Assessment]):
        self.catalog = catalog
        self.retriever = Retriever(catalog)

    def chat(self, messages: list[Message]) -> ChatResponse:
        state = build_state(messages)

        if state.off_scope or state.prompt_injection:
            return ChatResponse(reply=refusal_reply(), recommendations=[], end_of_conversation=False)

        if state.compare_targets:
            return self._compare(state.compare_targets)

        if needs_clarification(state):
            return ChatResponse(
                reply=clarification_question(state),
                recommendations=[],
                end_of_conversation=False,
            )

        scored = self.retriever.retrieve(state, top_k=40)
        recs = self._build_recommendations(scored, max_items=10)
        recs = validate_recommendations(recs, self.catalog)

        if not recs:
            return ChatResponse(
                reply="I could not find a safe catalog-grounded shortlist from the information given. Could you share the role, seniority and main skills to assess?",
                recommendations=[],
                end_of_conversation=False,
            )

        context_bits = []
        if state.role:
            context_bits.append(state.role)
        if state.seniority:
            context_bits.append(f"{state.seniority} level")
        if state.skills:
            context_bits.append(", ".join(state.skills[:4]))
        if state.soft_skills:
            context_bits.append(", ".join(state.soft_skills[:3]))

        context = " for " + "; ".join(context_bits) if context_bits else ""
        reply = f"Based on the SHL catalog, here are {len(recs)} assessment recommendations{context}."
        return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=True)

    def _build_recommendations(self, scored: list[ScoredAssessment], max_items: int) -> list[Recommendation]:
        recs: list[Recommendation] = []
        seen: set[str] = set()
        for row in scored:
            item = row.item
            if item.url in seen:
                continue
            if not item.test_type:
                continue
            seen.add(item.url)
            recs.append(Recommendation(name=item.name, url=item.url, test_type=item.test_type))
            if len(recs) >= max_items:
                break
        return recs

    def _compare(self, targets: list[str]) -> ChatResponse:
        if len(targets) < 2:
            return ChatResponse(
                reply="Which two SHL assessments should I compare?",
                recommendations=[],
                end_of_conversation=False,
            )

        left_matches = self.retriever.fuzzy_find(targets[0], limit=3)
        right_matches = self.retriever.fuzzy_find(targets[1], limit=3)

        if not left_matches or not right_matches:
            return ChatResponse(
                reply="I can compare SHL assessments, but I could not confidently match both names to catalog items. Please give the exact assessment names.",
                recommendations=[],
                end_of_conversation=False,
            )

        left = left_matches[0]
        right = right_matches[0]
        if left.url == right.url:
            return ChatResponse(
                reply=f"Both names appear to refer to {left.name}. Give me a second distinct SHL assessment to compare.",
                recommendations=[],
                end_of_conversation=False,
            )

        reply = self._comparison_reply(left, right)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=True)

    def _comparison_reply(self, a: Assessment, b: Assessment) -> str:
        def sentence(item: Assessment) -> str:
            label = TEST_TYPE_LABELS.get(item.test_type, item.test_type or "unknown type")
            duration = f", about {item.duration_minutes} minutes" if item.duration_minutes else ""
            desc = item.description.strip()
            if len(desc) > 220:
                desc = desc[:217].rstrip() + "..."
            return f"{item.name} is a {label} assessment{duration}. {desc}".strip()

        return (
            f"{sentence(a)}\n\n"
            f"{sentence(b)}\n\n"
            f"Main difference: {a.name} is categorized as {TEST_TYPE_LABELS.get(a.test_type, a.test_type)}, "
            f"while {b.name} is categorized as {TEST_TYPE_LABELS.get(b.test_type, b.test_type)}. "
            "So choose based on the construct you need to measure, not on the name alone."
        )
