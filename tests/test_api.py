from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_exact():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_vague_query_clarifies_without_recommendations():
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment"}]})
    body = r.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False


def test_java_query_recommends_catalog_items():
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "Hiring a mid-level Java developer with SQL and stakeholder communication"}]})
    body = r.json()
    assert 1 <= len(body["recommendations"]) <= 10
    assert any("Java" in rec["name"] for rec in body["recommendations"])
    assert all(rec["url"].startswith("https://www.shl.com/solutions/products/product-catalog/view/") for rec in body["recommendations"])


def test_off_scope_refuses():
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "Give me legal hiring advice"}]})
    body = r.json()
    assert body["recommendations"] == []
    assert "SHL" in body["reply"]


def test_refinement_adds_personality_without_resetting():
    r = client.post("/chat", json={"messages": [
        {"role": "user", "content": "Hiring a Java developer with SQL"},
        {"role": "assistant", "content": "What seniority?"},
        {"role": "user", "content": "Mid level. Actually add personality tests too."}
    ]})
    body = r.json()
    assert 1 <= len(body["recommendations"]) <= 10
    assert any(rec["test_type"] == "P" for rec in body["recommendations"])


def test_compare_grounded_no_recommendations():
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "What is the difference between OPQ and GSA?"}]})
    body = r.json()
    assert body["recommendations"] == []
    assert "OPQ" in body["reply"] or "OPQ32r" in body["reply"]
