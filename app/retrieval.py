from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable
from rapidfuzz import fuzz, process
try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - fallback for minimal environments
    BM25Okapi = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalog import Assessment, TEST_TYPE_LABELS
from .state import HiringState
from .text_utils import normalize_text, tokenize



class SimpleBM25:
    """Small BM25 fallback so the app remains runnable if rank-bm25 is absent."""

    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self.avgdl = sum(len(d) for d in tokenized_docs) / max(1, self.n_docs)
        self.df: dict[str, int] = {}
        self.tfs: list[dict[str, int]] = []
        for doc in tokenized_docs:
            tf: dict[str, int] = {}
            for tok in doc:
                tf[tok] = tf.get(tok, 0) + 1
            self.tfs.append(tf)
            for tok in tf:
                self.df[tok] = self.df.get(tok, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc, tf in zip(self.docs, self.tfs):
            dl = len(doc) or 1
            score = 0.0
            for tok in query_tokens:
                if tok not in tf:
                    continue
                df = self.df.get(tok, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                freq = tf[tok]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


@dataclass
class ScoredAssessment:
    item: Assessment
    score: float
    reasons: list[str]


class Retriever:
    def __init__(self, catalog: list[Assessment]):
        self.catalog = catalog
        self.docs = [item.search_text for item in catalog]
        self.tokenized_docs = [tokenize(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs) if BM25Okapi else SimpleBM25(self.tokenized_docs)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.tfidf = self.vectorizer.fit_transform(self.docs)
        self.name_choices = {item.name: item for item in catalog}
        self.alias_choices: dict[str, Assessment] = {}
        for item in catalog:
            self.alias_choices[item.name.lower()] = item
            for alias in item.aliases:
                self.alias_choices[alias.lower()] = item

    def fuzzy_find(self, text: str, limit: int = 5) -> list[Assessment]:
        query = normalize_text(text)
        if not query:
            return []
        choices = list(self.alias_choices.keys())
        matches = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)
        out: list[Assessment] = []
        seen: set[str] = set()
        for matched_name, score, _ in matches:
            if score < 72:
                continue
            item = self.alias_choices[matched_name]
            if item.url in seen:
                continue
            seen.add(item.url)
            out.append(item)
        return out

    def retrieve(self, state: HiringState, top_k: int = 30) -> list[ScoredAssessment]:
        query = self._expanded_query(state)
        if not query:
            query = state.raw_user_text

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_norm = self._normalize(bm25_scores)

        q_vec = self.vectorizer.transform([query])
        dense_scores = cosine_similarity(q_vec, self.tfidf).flatten()
        dense_norm = self._normalize(dense_scores)

        scored: list[ScoredAssessment] = []
        for idx, item in enumerate(self.catalog):
            meta_score, reasons = self._metadata_score(item, state)
            score = 0.45 * float(bm25_norm[idx]) + 0.35 * float(dense_norm[idx]) + meta_score
            if item.test_type in state.excluded_test_types:
                score -= 10.0
                reasons.append(f"excluded type {item.test_type}")
            scored.append(ScoredAssessment(item=item, score=score, reasons=reasons))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _expanded_query(self, state: HiringState) -> str:
        expansions: list[str] = [state.query_text()]
        for code in state.requested_test_types:
            label = TEST_TYPE_LABELS.get(code)
            if label:
                expansions.append(label)
        for skill in state.skills:
            expansions.extend(self._skill_expansions(skill))
        for soft in state.soft_skills:
            expansions.extend(self._soft_expansions(soft))
        return " ".join(expansions)

    @staticmethod
    def _skill_expansions(skill: str) -> list[str]:
        mapping = {
            "java": ["Java Core Java Java 8 programming coding developer"],
            "python": ["Python programming coding developer software"],
            "javascript": ["JavaScript Node React web developer frontend"],
            "sql": ["SQL database queries data analyst developer"],
            "excel": ["Microsoft Excel spreadsheet data analyst"],
            "data science": ["statistics machine learning analytics data science econometrics"],
            "finance": ["financial accounting bookkeeping audit numerical"],
            "sales": ["sales customer persuasion negotiation"],
            "customer service": ["customer service contact center call center support simulation"],
        }
        return mapping.get(skill, [skill])

    @staticmethod
    def _soft_expansions(skill: str) -> list[str]:
        mapping = {
            "communication": ["communication stakeholder interpersonal workplace behavior personality"],
            "teamwork": ["teamwork collaboration interpersonal workplace behavior personality"],
            "leadership": ["leadership manager management competencies personality"],
            "personality": ["personality behavior OPQ workplace style"],
            "situational judgement": ["situational judgement scenario biodata"],
        }
        return mapping.get(skill, [skill])

    def _metadata_score(self, item: Assessment, state: HiringState) -> tuple[float, list[str]]:
        text = normalize_text(item.search_text)
        name = normalize_text(item.name)
        score = 0.0
        reasons: list[str] = []

        if state.role and state.role in text:
            score += 0.5
            reasons.append("role match")

        for skill in state.skills:
            if skill in name:
                score += 3.0
                reasons.append(f"exact skill in name: {skill}")
            elif skill in text:
                score += 1.2
                reasons.append(f"skill match: {skill}")

        for soft in state.soft_skills:
            if soft in text:
                score += 0.9
                reasons.append(f"soft skill match: {soft}")
            if soft in {"communication", "teamwork", "personality"} and item.test_type in {"P", "B", "C"}:
                score += 0.8
                reasons.append("behavioral type fit")
            if soft == "leadership" and item.test_type in {"P", "C", "A"}:
                score += 0.8
                reasons.append("leadership type fit")

        for code in state.requested_test_types:
            if item.test_type == code:
                score += 1.6
                reasons.append(f"requested type {code}")

        # Developer / technical role defaults.
        role_text = normalize_text(state.role or "")
        if any(term in role_text for term in ["developer", "engineer", "software", "qa", "tester"]):
            if item.test_type == "K":
                score += 0.9
                reasons.append("technical role type fit")
            if any(skill in text for skill in ["programming", "java", "python", "sql", "coding", "software"]):
                score += 0.8
                reasons.append("technical role text fit")

        if any(term in role_text for term in ["manager", "leader", "supervisor"]):
            if item.test_type in {"P", "C", "A"}:
                score += 0.8
                reasons.append("manager type fit")

        if any(term in role_text for term in ["graduate", "entry"]):
            if item.test_type in {"A", "K", "P"}:
                score += 0.4
                reasons.append("graduate type fit")

        # Prefer specific product names over very generic ones when a hard skill is present.
        if state.skills and item.test_type in {"A", "P"} and not state.soft_skills and item.test_type not in state.requested_test_types:
            score -= 0.4

        return score, reasons

    @staticmethod
    def _normalize(values: Iterable[float]):
        vals = list(values)
        if not vals:
            return []
        lo = min(vals)
        hi = max(vals)
        if math.isclose(lo, hi):
            return [0.0 for _ in vals]
        return [(v - lo) / (hi - lo) for v in vals]
