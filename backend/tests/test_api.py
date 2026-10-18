"""API smoke tests against the local database (run after `bootstrap`)."""
import os

import pytest

os.environ.setdefault("IPOPILOT_ENABLE_SCHEDULER", "0")
from fastapi.testclient import TestClient  # noqa: E402

from ipopilot.api.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_allotment_calc_matches_formula(client):
    r = client.post("/api/allotment/calc", json={"board": "sme", "category": "RII", "subscription_times": 10, "applicants": 3})
    d = r.json()
    assert d["p_mean"] == pytest.approx(0.10, abs=1e-6)  # SME individual: L = 1 exactly
    assert d["table"][2]["p_at_least_one"] == pytest.approx(0.271, abs=1e-3)


def test_allotment_calc_exact_mode(client):
    r = client.post("/api/allotment/calc", json={"subscription_times": 50, "applications": 100000,
                                                 "offered_shares": 300000, "unit_shares": 30})
    assert r.json()["p_mean"] == pytest.approx(0.10)


def test_plan_validation(client):
    assert client.post("/api/plan", json={"capital": -5, "applicants": [{"name": "a"}]}).status_code == 422
    assert client.post("/api/plan", json={"capital": 1000, "applicants": []}).status_code == 422


def test_plan_with_demo_and_duplicate_pan(client):
    r = client.post("/api/plan", json={"capital": 100000, "include_demo": True, "objective": "compare",
                                       "applicants": [{"name": "A", "pan": "X"}, {"name": "B", "pan": "X"}]})
    assert r.status_code == 200
    d = r.json()
    assert any("shares a PAN" in w for w in d["warnings"])
    assert len(d["comparison"]) == 4
    for plan in d["plans"].values():
        assert plan["peak_capital_blocked"] <= 100000 + 1e-6
        for row in plan["selected"]:
            assert len({a["applicant_id"] for a in row["applications"]}) == len(row["applications"])


def test_ipo_detail_and_whatif(client):
    base = client.get("/api/ipos/demo-aurora-robotics").json()
    low = client.post("/api/ipos/demo-aurora-robotics/whatif", json={"sub_rii": 1.0, "final_subscription": True}).json()
    assert base["distribution"] and base["options"]
    # an undersubscribed retail book means near-certain allotment
    assert low["allotment"]["p_mean"] > 0.95
