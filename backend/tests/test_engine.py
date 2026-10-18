import datetime as dt

import numpy as np
import pytest

from ipopilot.engine.allotment import (allot_given, application_options, naive_at_least_one, p_allot_point,
                                       prob_at_least_one)
from ipopilot.engine.optimizer import Applicant, Candidate, PlanRequest, build_plan, compare_objectives
from ipopilot.engine.subscription import elapsed_fraction
from ipopilot.ingestion.util import name_similarity, parse_band, parse_day_month


# ---------------------------------------------------------------- allotment maths
def test_example_from_spec_10pct():
    # 1,00,000 applications, 10,000 lots -> 10%; with L=1 the subscription is 10x
    p, units = allot_given(np.array([10.0]), np.array([1.0]), 1)
    assert p[0] == pytest.approx(0.10)
    assert units[0] == 1


def test_at_least_one_naive_matches_spec_table():
    assert naive_at_least_one(0.10, 3) == pytest.approx(0.271)
    assert naive_at_least_one(0.10, 5) == pytest.approx(0.40951)


def test_shared_uncertainty_lowers_multi_application_odds():
    p = np.array([0.0, 0.2])  # same mean 0.1 but uncertain
    assert prob_at_least_one(p, 3) < naive_at_least_one(0.1, 3)


def test_undersubscribed_full_allotment():
    p, units = allot_given(np.array([0.6]), np.array([1.7]), 5)
    assert p[0] == 1.0 and units[0] == 5


def test_more_lots_do_not_raise_odds_when_oversubscribed():
    p1, _ = allot_given(np.array([40.0]), np.array([1.7]), 1)
    p5, u5 = allot_given(np.array([40.0]), np.array([1.7]), 5)
    assert p1[0] == p5[0]
    assert u5[0] == 1  # lottery winners get exactly one lot


def test_proportional_remainder_when_everyone_gets_one():
    p, units = allot_given(np.array([1.5]), np.array([3.0]), 4)
    assert p[0] == 1.0
    assert 1 < units[0] < 4


def test_exact_applications_override():
    assert p_allot_point("mainboard", "RII", 50, applications=100_000, offered_shares=300_000, unit_shares=30) == pytest.approx(0.1)


def test_mainboard_options_respect_2_lakh():
    opts = application_options("mainboard", 500, 30, dt.date(2026, 1, 1))
    rii = [o for o in opts if o.category == "RII"]
    assert all(o.amount <= 200_000 for o in rii)
    snii = [o for o in opts if o.category == "SNII"][0]
    assert snii.amount > 200_000
    bnii = [o for o in opts if o.category == "BNII"][0]
    assert bnii.amount > 1_000_000


def test_sme_two_lot_minimum_after_july_2025():
    new = application_options("sme", 100, 1200, dt.date(2025, 9, 1))
    old = application_options("sme", 100, 1200, dt.date(2024, 9, 1))
    assert new[0].lots == 2 and old[0].lots == 1


# ---------------------------------------------------------------- optimizer
def _cand(slug, mean, T, price=500, lot=30, close=dt.date(2030, 1, 7), board="mainboard", S=3000, seed=0):
    rng = np.random.default_rng(seed)
    r = mean + 0.05 * rng.standard_normal(S)
    Ts = np.full(S, T)
    return Candidate(slug, slug, board, price, lot, close, close + dt.timedelta(days=1), r,
                     {"RII": Ts, "SNII": Ts, "BNII": Ts}, float((r < 0).mean()), float(r.mean()))


def test_one_application_per_pan_and_budget():
    cands = [_cand("a", 0.3, 20), _cand("b", 0.2, 5, seed=1)]
    req = PlanRequest(capital=40_000, applicants=[Applicant("0", "A", "P1"), Applicant("1", "B", "P2")],
                      objective="max_ev", allow_nii=False)
    plan = build_plan(cands, req)
    assert plan["peak_capital_blocked"] <= 40_000
    for row in plan["selected"]:
        ids = [a["applicant_id"] for a in row["applications"]]
        assert len(ids) == len(set(ids))


def test_negative_ev_excluded():
    plan = build_plan([_cand("bad", -0.2, 5)], PlanRequest(capital=100_000, applicants=[Applicant("0", "A", "P")]))
    assert plan["selected"] == [] and plan["excluded"]


def test_capital_reuse_across_non_overlapping_windows():
    c1 = _cand("a", 0.3, 2, close=dt.date(2030, 1, 7))
    c2 = _cand("b", 0.3, 2, close=dt.date(2030, 1, 14), seed=2)
    req = PlanRequest(capital=15_000, applicants=[Applicant("0", "A", "P")], allow_nii=False)
    plan = build_plan([c1, c2], req)
    assert len(plan["selected"]) == 2  # same ₹15k used twice


def test_compare_runs_all_objectives():
    cands = [_cand("a", 0.3, 30), _cand("b", 0.1, 3, seed=1), _cand("c", 0.05, 1.5, seed=2)]
    res = compare_objectives(cands, PlanRequest(capital=100_000, applicants=[Applicant("0", "A", "P1"),
                                                                             Applicant("1", "B", "P2")]))
    assert {r["objective"] for r in res["comparison"]} == {"max_ev", "max_allot_prob", "capital_efficiency", "risk_adjusted"}
    allot = next(r for r in res["comparison"] if r["objective"] == "max_allot_prob")
    ev = next(r for r in res["comparison"] if r["objective"] == "max_ev")
    assert allot["p_any_allotment"] >= ev["p_any_allotment"] - 1e-6


# ---------------------------------------------------------------- helpers
def test_elapsed_fraction():
    o, c = dt.date(2026, 10, 15), dt.date(2026, 10, 19)  # Thu..Mon = 3 trading days
    assert elapsed_fraction(o, c, dt.datetime(2026, 10, 15, 9, 0)) == 0
    assert elapsed_fraction(o, c, dt.datetime(2026, 10, 16, 17, 0)) == pytest.approx(2 / 3)
    assert elapsed_fraction(o, c, dt.datetime(2026, 10, 20, 9, 0)) == 1


def test_parsers():
    assert parse_band("Rs.77 to Rs.82") == (77, 82)
    assert parse_band("558.00 - 586.00") == (558, 586)
    assert parse_day_month("6 May", 2025, dt.date(2025, 5, 1)) == dt.date(2025, 5, 6)
    assert parse_day_month("30 December", 2026, dt.date(2026, 1, 2)) == dt.date(2025, 12, 30)
    assert name_similarity("Tonbo Imaging", "Tonbo Imaging India Limited") > 0.86
