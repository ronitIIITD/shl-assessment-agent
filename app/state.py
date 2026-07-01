import re
from pydantic import BaseModel, Field
from .schemas import Message
from .text_utils import normalize_text


SKILL_SYNONYMS: dict[str, list[str]] = {
    "java": ["java", "j2ee", "spring", "springboot", "spring boot"],
    "python": ["python", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "node", "node.js", "react", "angular", "typescript"],
    "sql": ["sql", "mysql", "postgres", "postgresql", "database", "query"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp", ".net", "dotnet"],
    "excel": ["excel", "spreadsheet", "microsoft excel"],
    "data science": ["data science", "machine learning", "ml", "analytics", "statistical", "statistics"],
    "devops": ["devops", "aws", "azure", "docker", "kubernetes", "cloud"],
    "customer service": ["customer service", "contact center", "call center", "support"],
    "sales": ["sales", "account executive", "business development"],
    "finance": ["finance", "financial", "accounting", "bookkeeping", "audit"],
    "numerical reasoning": ["numerical", "quantitative", "numbers", "data interpretation"],
    "verbal reasoning": ["verbal", "reading comprehension", "written information"],
    "deductive reasoning": ["deductive", "logical reasoning", "logic"],
    "inductive reasoning": ["inductive", "pattern", "abstract reasoning"],
}

SOFT_SKILLS: dict[str, list[str]] = {
    "communication": ["communication", "communicate", "stakeholder", "client-facing", "client facing", "presentation"],
    "teamwork": ["teamwork", "collaboration", "collaborate", "team", "cross functional", "cross-functional"],
    "leadership": ["leadership", "lead", "manager", "management", "supervise", "supervisor"],
    "personality": ["personality", "behavior", "behaviour", "work style", "culture fit", "opq"],
    "situational judgement": ["situational", "judgement", "judgment", "sjt", "scenario"],
}

ROLE_WORDS = [
    "developer", "engineer", "analyst", "manager", "sales", "support", "graduate",
    "consultant", "accountant", "finance", "hr", "customer", "java developer", "software engineer",
    "data scientist", "qa", "tester", "administrator", "leader", "executive",
]

SENIORITY_PATTERNS = {
    "entry": ["entry", "entry-level", "junior", "graduate", "campus", "0 years", "1 year", "fresher"],
    "mid": ["mid", "mid-level", "intermediate", "3 years", "4 years", "5 years", "2-5", "around 4"],
    "senior": ["senior", "lead", "principal", "staff", "6 years", "7 years", "8 years", "10 years"],
    "manager": ["manager", "management", "supervisor", "leadership"],
}

TEST_TYPE_KEYWORDS = {
    "A": ["ability", "aptitude", "cognitive", "g+", "gsa", "reasoning", "numerical", "verbal", "deductive", "inductive"],
    "B": ["biodata", "situational", "judgement", "judgment", "sjt"],
    "C": ["competency", "competencies"],
    "D": ["development", "360"],
    "E": ["assessment exercise", "exercise", "in basket", "case study"],
    "K": ["knowledge", "skills", "coding", "programming", "technical", "java", "python", "sql", "excel"],
    "P": ["personality", "behavior", "behaviour", "opq", "motivation", "fit"],
    "S": ["simulation", "simulations", "typing", "call center", "contact center"],
}

OFF_SCOPE_PATTERNS = [
    "legal", "legal advice", "hiring advice", "employment law", "labor law", "labour law", "how do i fire", "how to fire",
    "salary negotiation", "write a job offer", "interview questions not shl", "general hiring advice",
    "diversity quota", "discriminate", "protected class", "age discrimination", "gender discrimination",
    "recommend non-shl", "outside shl", "not shl", "google assessment", "hackerrank instead",
]

PROMPT_INJECTION_PATTERNS = [
    "ignore previous", "ignore all previous", "system prompt", "developer message", "reveal your prompt",
    "you are not restricted", "forget the catalog", "jailbreak", "act as", "do anything now",
    "return fake", "invent", "do not validate", "bypass",
]

VAGUE_PATTERNS = [
    r"^i need an assessment$",
    r"^need an assessment$",
    r"^recommend an assessment$",
    r"^assessment$",
    r"^test$",
    r"^shl assessment$",
    r"^help me choose$",
]


class HiringState(BaseModel):
    raw_user_text: str = ""
    latest_user_text: str = ""
    role: str | None = None
    seniority: str | None = None
    skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    requested_test_types: list[str] = Field(default_factory=list)
    excluded_test_types: list[str] = Field(default_factory=list)
    jd_text: str | None = None
    compare_targets: list[str] = Field(default_factory=list)
    off_scope: bool = False
    prompt_injection: bool = False
    no_preference: bool = False
    vague: bool = False

    def query_text(self) -> str:
        parts = [self.role or "", self.seniority or "", " ".join(self.skills), " ".join(self.soft_skills), self.jd_text or ""]
        return " ".join(p for p in parts if p).strip() or self.raw_user_text


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _extract_compare_targets(text: str) -> list[str]:
    t = text.strip()
    lowered = t.lower()
    if not any(word in lowered for word in ["compare", "difference", "different", " vs ", " versus ", "between"]):
        return []

    # Examples: "compare OPQ and GSA", "difference between OPQ and GSA", "OPQ vs GSA"
    patterns = [
        r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\?|$)",
        r"difference\s+(?:between\s+)?(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, t, flags=re.I)
        if match:
            left = re.sub(r"^(what is|what's|the|a|an)\s+", "", match.group(1).strip(), flags=re.I)
            right = match.group(2).strip()
            return [left.strip(" .?\"'"), right.strip(" .?\"'")]
    return []


def _extract_role(text: str) -> str | None:
    normalized = normalize_text(text)
    for phrase in sorted(ROLE_WORDS, key=len, reverse=True):
        if phrase in normalized:
            return phrase

    # Catch common JD phrasing.
    match = re.search(r"hiring\s+(?:a|an)?\s*([a-z0-9+#.\- ]{2,60})", text, flags=re.I)
    if match:
        candidate = match.group(1).split(" who ")[0].split(" with ")[0].strip(" .")
        if candidate and "advice" not in candidate.lower():
            return candidate.lower()
    return None


def _extract_seniority(text: str) -> str | None:
    normalized = normalize_text(text)
    for level, patterns in SENIORITY_PATTERNS.items():
        if _contains_any(normalized, patterns):
            return level
    years = re.search(r"(\d+)\+?\s*(?:years|yrs)", normalized)
    if years:
        n = int(years.group(1))
        if n <= 2:
            return "entry"
        if n <= 5:
            return "mid"
        return "senior"
    return None


def _extract_skills(text: str) -> list[str]:
    normalized = normalize_text(text)
    skills: list[str] = []
    for canonical, variants in SKILL_SYNONYMS.items():
        if _contains_any(normalized, variants):
            _append_unique(skills, canonical)
    return skills


def _extract_soft_skills(text: str) -> list[str]:
    normalized = normalize_text(text)
    out: list[str] = []
    for canonical, variants in SOFT_SKILLS.items():
        if _contains_any(normalized, variants):
            _append_unique(out, canonical)
    return out


def _extract_test_type_preferences(text: str) -> tuple[list[str], list[str]]:
    normalized = normalize_text(text)
    requested: list[str] = []
    excluded: list[str] = []

    for code, keywords in TEST_TYPE_KEYWORDS.items():
        hit_keywords = [kw for kw in keywords if kw in normalized]
        if not hit_keywords:
            continue
        exclusion_hit = False
        for kw in hit_keywords:
            # Look for local negation, not global negation.
            if re.search(rf"\b(no|not|without|exclude|remove|avoid)\b[^.\n]{{0,35}}\b{re.escape(kw)}\b", normalized):
                exclusion_hit = True
                break
        if exclusion_hit:
            _append_unique(excluded, code)
        else:
            _append_unique(requested, code)

    return requested, excluded


def build_state(messages: list[Message]) -> HiringState:
    user_texts = [m.content for m in messages if m.role == "user"]
    latest = user_texts[-1] if user_texts else ""
    combined = "\n".join(user_texts)
    normalized = normalize_text(combined)
    latest_norm = normalize_text(latest)

    state = HiringState(raw_user_text=combined, latest_user_text=latest)
    state.off_scope = _contains_any(normalized, OFF_SCOPE_PATTERNS)
    state.prompt_injection = _contains_any(normalized, PROMPT_INJECTION_PATTERNS)
    state.no_preference = any(p in latest_norm for p in ["no preference", "no prefs", "not sure", "doesn't matter", "dont know", "don't know"])
    state.vague = any(re.match(pattern, latest_norm) for pattern in VAGUE_PATTERNS)

    state.compare_targets = _extract_compare_targets(latest)
    state.role = _extract_role(combined)
    state.seniority = _extract_seniority(combined)
    state.skills = _extract_skills(combined)
    state.soft_skills = _extract_soft_skills(combined)
    req, exc = _extract_test_type_preferences(combined)
    state.requested_test_types = req
    state.excluded_test_types = exc

    # Treat long messages and JD-like language as a job description.
    if len(latest) > 250 or any(marker in latest_norm for marker in ["job description", "responsibilities", "requirements", "qualifications"]):
        state.jd_text = latest
    elif len(combined) > 500:
        state.jd_text = combined[-4000:]

    return state


def needs_clarification(state: HiringState) -> bool:
    if state.off_scope or state.prompt_injection or state.compare_targets:
        return False
    if state.no_preference:
        return False
    if state.jd_text:
        return False
    if state.vague:
        return True

    has_role = bool(state.role)
    has_skill = bool(state.skills or state.soft_skills or state.requested_test_types)

    # A single concrete technical skill is enough to retrieve. A role alone is often too broad.
    if has_role and has_skill:
        return False
    if state.skills and len(state.raw_user_text) > 20:
        return False
    return True


def clarification_question(state: HiringState) -> str:
    if not state.role:
        return "What role are you hiring for, and what should the assessment focus on: technical skill, cognitive ability, personality, simulations or communication?"
    return f"For the {state.role} role, what should I prioritize: technical skills, cognitive ability, personality/behavior, simulations or communication?"
