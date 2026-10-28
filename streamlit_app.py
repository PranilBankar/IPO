"""IPOPilot AI — Streamlit Web Application
An interactive deep-learning intelligence and decision-support platform for Indian IPOs.
Supports direct deployment to Streamlit Community Cloud and local execution.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import sys
from typing import Any

# Ensure backend package is in python search path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import ipopilot backend modules with graceful resilience
try:
    from ipopilot.config import settings
    from ipopilot.db import GmpObservation, Ipo, SubscriptionSnapshot, init_db, session_scope
    from ipopilot.engine import service
    from ipopilot.engine.allotment import allot_given, prob_at_least_one, naive_at_least_one, sample_L
    from ipopilot.ingestion.pipeline import ensure_demo_ipo, run_all_live
    BACKEND_AVAILABLE = True
except Exception as exc:
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Page configuration & Modern Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IPOPilot AI — Deep Learning IPO Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 34, 52, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.5);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-main { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-sme { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-open { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-closed { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }
    .badge-demo { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

    /* Custom Callouts */
    .callout-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin: 10px 0;
    }
    .callout-warning {
        border-left-color: #f59e0b;
        background: rgba(40, 25, 15, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Backend Bootstrapping & Data Fetching Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def bootstrap_database():
    """Initializes SQLite database and ensures fictional demo data is populated."""
    if not BACKEND_AVAILABLE:
        return False
    try:
        init_db()
        ensure_demo_ipo()
        return True
    except Exception as e:
        st.warning(f"Note on DB Initialization: {e}")
        return False


def format_inr(val: float | None) -> str:
    """Format rupees with commas and symbol."""
    if val is None or math.isnan(val):
        return "—"
    if abs(val) >= 1e7:
        return f"₹{val / 1e7:.2f} Cr"
    if abs(val) >= 1e5:
        return f"₹{val / 1e5:.2f} L"
    return f"₹{val:,.0f}"


def format_pct(val: float | None, sign: bool = True) -> str:
    if val is None or math.isnan(val):
        return "—"
    prefix = "+" if (sign and val > 0) else ""
    return f"{prefix}{val:.1f}%"


# Run DB bootstrap
db_ready = bootstrap_database()

# ---------------------------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 2rem;">🚀</span>
            <div>
                <h2 style="margin: 0; font-size: 1.3rem; font-weight: 800; background: linear-gradient(90deg, #60a5fa, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">IPOPilot AI</h2>
                <span style="font-size: 0.75rem; color: #94a3b8;">Deep Learning IPO Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Live IPO Intelligence",
            "🔬 IPO Deep Dive & What-If",
            "🎯 Capital Optimizer Planner",
            "🧮 Allotment Odds Calculator",
            "📜 Historical Data & Backtest",
            "🧠 AI Model Architecture",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Filters & Data")
    board_filter = st.selectbox("Market Board", ["All Boards", "Mainboard Only", "SME Only"], index=0)
    include_demo = st.checkbox("Include Demo Issue (Aurora Robotics)", value=True, help="Includes fictional benchmark issue for interface exploration.")

    st.markdown("---")
    # Quick health & actions
    if BACKEND_AVAILABLE:
        try:
            m_version = service.model_version()
            st.caption(f"🤖 **Model Engine:** `{m_version}`")
        except Exception:
            st.caption("🤖 **Model Engine:** `Active`")

        col_sync, col_clear = st.columns(2)
        with col_sync:
            if st.button("🔄 Sync Live", help="Fetch latest live subscriptions and GMP from NSE / IPOWatch"):
                with st.spinner("Scraping live sources..."):
                    try:
                        res = run_all_live()
                        st.toast("Sync complete!", icon="✅")
                    except Exception as err:
                        st.error(f"Sync error: {err}")
        with col_clear:
            if st.button("🧹 Cache", help="Clear UI cache"):
                st.cache_data.clear()
                st.toast("Cache refreshed!")

    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748b; line-height: 1.4; margin-top: 15px;">
            <b>Disclaimer:</b> Educational decision-support tool. Not SEBI-registered investment advice. GMP is unofficial & volatile.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data Loaders (Cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_active_ipos(demo: bool):
    if not BACKEND_AVAILABLE:
        return []
    try:
        return service.list_active(include_demo=demo)
    except Exception as e:
        st.error(f"Error fetching active IPOs: {e}")
        return []


@st.cache_data(ttl=120)
def get_history_ipos():
    if not BACKEND_AVAILABLE:
        return pd.DataFrame()
    try:
        with session_scope() as s:
            from sqlalchemy import select
            rows = s.execute(
                select(
                    Ipo.slug,
                    Ipo.name,
                    Ipo.symbol,
                    Ipo.board,
                    Ipo.close_date,
                    Ipo.listing_date,
                    Ipo.price_high,
                    Ipo.issue_price,
                    Ipo.issue_size_cr,
                    Ipo.listing_gain_pct,
                    Ipo.sector,
                ).where(Ipo.listing_gain_pct.is_not(None), Ipo.is_demo.is_(False))
            ).all()

            data = []
            for r in rows:
                data.append({
                    "slug": r.slug,
                    "name": r.name,
                    "symbol": r.symbol or "",
                    "board": r.board,
                    "close_date": r.close_date,
                    "listing_date": r.listing_date,
                    "issue_price": r.issue_price or r.price_high,
                    "issue_size_cr": r.issue_size_cr,
                    "listing_gain_pct": r.listing_gain_pct,
                    "sector": r.sector or "General",
                })
            df = pd.DataFrame(data)
            if not df.empty and "listing_date" in df.columns:
                df["year"] = pd.to_datetime(df["listing_date"]).dt.year
            return df
    except Exception as e:
        st.error(f"Error loading historical issues: {e}")
        return pd.DataFrame()


# ===========================================================================
# PAGE 1: 📊 LIVE IPO INTELLIGENCE
# ===========================================================================
if page == "📊 Live IPO Intelligence":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">📊 Live IPO Intelligence Dashboard</h1>
            <p style="margin: 5px 0 0; color: #94a3b8; font-size: 0.95rem;">
                Live Indian Mainboard & SME issues with verified subscription data, unofficial grey market premiums (GMP),
                Deep Mixture Density Network (MDN) listing predictions, and SEBI-lottery allotment probability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    items = get_active_ipos(include_demo)
    if board_filter == "Mainboard Only":
        items = [i for i in items if i.get("board") == "mainboard"]
    elif board_filter == "SME Only":
        items = [i for i in items if i.get("board") == "sme"]

    open_items = [i for i in items if i.get("stage") == "open"]
    best_ev_item = None
    if open_items:
        valid_evs = [i for i in open_items if (i.get("min_application_ev") or {}).get("ev_per_rupee") is not None]
        if valid_evs:
            best_ev_item = max(valid_evs, key=lambda x: x["min_application_ev"]["ev_per_rupee"])

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Open for Bidding</div>
                <div class="metric-value" style="color: #34d399;">{len(open_items)}</div>
                <div class="metric-sub">{len(items)} total active / upcoming</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        top_name = (best_ev_item["name"][:20] + "...") if best_ev_item else "No open issues"
        top_val = format_pct(best_ev_item["min_application_ev"]["ev_per_rupee"] * 100) if best_ev_item else "—"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Top Expected Return / ₹</div>
                <div class="metric-value" style="color: #60a5fa;">{top_val}</div>
                <div class="metric-sub">{top_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        m_version = service.model_version() if BACKEND_AVAILABLE else "v1.0"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">AI Predictive Model</div>
                <div class="metric-value" style="font-size: 1.3rem; line-height: 2rem; color: #c084fc;">{m_version}</div>
                <div class="metric-sub">Deep MDN Ensemble (5 members)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Live Feeds</div>
                <div class="metric-value" style="color: #fbbf24;">Synced</div>
                <div class="metric-sub">NSE, BSE, IPOWatch, News</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 Active & Upcoming Issue Board")

    if not items:
        st.info("No active IPOs found for the selected filter. Try enabling 'Include Demo Issue' in the sidebar.")
    else:
        # Construct summary table
        rows_display = []
        for r in items:
            pred = r.get("prediction") or {}
            sub = r.get("subscription") or {}
            ev = r.get("min_application_ev") or {}
            gmp_rs = (r.get("gmp_latest") or {}).get("gmp_rs")
            gmp_pct = r.get("gmp_consensus_pct")

            p_mean = pred.get("mean_pct")
            q10 = pred.get("q10_pct")
            q90 = pred.get("q90_pct")
            pred_range = f"[{q10:.0f}% to {q90:.0f}%]" if (q10 is not None and q90 is not None) else "—"

            rows_display.append({
                "IPO Name": r.get("name"),
                "Board": "Mainboard" if r.get("board") == "mainboard" else "SME",
                "Stage": (r.get("stage") or "").upper(),
                "Dates": f"{r.get('open_date') or ''} → {r.get('close_date') or ''}",
                "Price (₹)": f"₹{r.get('price') or r.get('price_high') or '—'}",
                "Min ₹": format_inr(r.get("min_investment")),
                "Total Sub": f"{sub.get('total'):.2f}x" if sub.get("total") is not None else "—",
                "GMP (₹)": f"₹{gmp_rs:.0f} ({gmp_pct:.1f}%)" if gmp_rs is not None and gmp_pct is not None else "—",
                "AI Expected Pop": format_pct(p_mean),
                "80% Conf. Range": pred_range,
                "P(Loss)": f"{pred.get('p_loss', 0)*100:.1f}%" if pred.get("p_loss") is not None else "—",
                "Allotment Odds": f"{r.get('allotment', {}).get('p_mean', 0)*100:.1f}%" if r.get("allotment") else "—",
                "Expected Profit (1 lot)": format_inr(ev.get("ev_rupees")),
                "slug": r.get("slug"),
            })

        df_display = pd.DataFrame(rows_display)
        st.dataframe(
            df_display.drop(columns=["slug"]),
            use_container_width=True,
            column_config={
                "IPO Name": st.column_config.TextColumn(width="medium"),
                "AI Expected Pop": st.column_config.TextColumn(width="small"),
                "Expected Profit (1 lot)": st.column_config.TextColumn(width="small"),
            },
            hide_index=True,
        )

        st.caption("💡 *Tip: Head over to **IPO Deep Dive & What-If** to simulate hypothetical subscriptions or check detailed predictive probability curves.*")

    # Educational Alerts
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown(
            """
            <div class="callout-box">
                <b>Why Minimum Application?</b><br/>
                Under SEBI retail rules, in an oversubscribed issue, successful bidders receive exactly ONE minimum lot by lottery. Applying for multiple lots blocks additional capital without improving lottery odds.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_e2:
        st.markdown(
            """
            <div class="callout-box">
                <b>The Winner's Curse:</b><br/>
                Investors are most likely to receive allotment when demand is weakest. IPOPilot couples subscription demand with listing returns via copulas so expected profit is realistically adjusted.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_e3:
        st.markdown(
            """
            <div class="callout-box callout-warning">
                <b>GMP Reliability:</b><br/>
                Grey market premiums are unofficial dealer indications and can be manipulated or stale. The ensemble model weights fundamentals and subscription velocity alongside GMP.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 2: 🔬 IPO DEEP DIVE & WHAT-IF SIMULATOR
# ===========================================================================
elif page == "🔬 IPO Deep Dive & What-If":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">🔬 IPO Deep Dive & Interactive Simulation Studio</h1>
            <p style="margin: 5px 0 0; color: #94a3b8; font-size: 0.95rem;">
                Explore complete predictive distributions, category subscription velocity, GMP history, and test "What-If" scenarios.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    items = get_active_ipos(include_demo)
    if not items:
        st.warning("No active IPOs available.")
    else:
        ipo_slug_map = {f"{i['name']} ({i['board'].upper()})": i["slug"] for i in items}
        selected_label = st.selectbox("Select IPO for Deep Analysis", list(ipo_slug_map.keys()), index=0)
        selected_slug = ipo_slug_map[selected_label]

        # Fetch full detail
        detail = service.ipo_detail(selected_slug)
        if not detail:
            st.error("Failed to load details for the selected issue.")
        else:
            # Header Cards
            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("Issue Price", f"₹{detail.get('price') or detail.get('price_high') or '—'}")
            h2.metric("Lot Size", f"{detail.get('lot_size') or '—'} shares")
            h3.metric("Min Capital Blocked", format_inr(detail.get("min_investment")))
            h4.metric("Issue Size", format_inr((detail.get("issue_size_cr") or 0) * 1e7))
            h5.metric("Sector", detail.get("sector") or "General")

            st.markdown("---")

            # Main Tabs for the deep dive
            tab_overview, tab_whatif, tab_data = st.tabs(["📈 Model Predictions & Distribution", "🎛️ Interactive 'What-If' Studio", "📑 Historical Data & News"])

            with tab_overview:
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown("#### 🎯 AI Listing Day Return Distribution")
                    pred = detail.get("prediction") or {}
                    dist = detail.get("distribution") or []

                    if dist:
                        df_dist = pd.DataFrame(dist)
                        # Plot density curve
                        fig_dist = px.area(
                            df_dist,
                            x="ret_pct",
                            y="density",
                            title=f"Predictive Probability Density: {detail.get('name')}",
                            labels={"ret_pct": "Listing Day Return (%)", "density": "Probability Density"},
                            color_discrete_sequence=["#3b82f6"],
                        )
                        # Add vertical indicators for key percentiles
                        fig_dist.add_vline(x=0, line_dash="dash", line_color="#ef4444", annotation_text="0% (Break-even)")
                        if pred.get("mean_pct") is not None:
                            fig_dist.add_vline(x=pred["mean_pct"], line_color="#10b981", annotation_text=f"Mean ({pred['mean_pct']:.1f}%)")
                        if pred.get("q10_pct") is not None:
                            fig_dist.add_vline(x=pred["q10_pct"], line_dash="dot", line_color="#94a3b8", annotation_text="Q10")
                        if pred.get("q90_pct") is not None:
                            fig_dist.add_vline(x=pred["q90_pct"], line_dash="dot", line_color="#94a3b8", annotation_text="Q90")

                        fig_dist.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15, 23, 42, 0.6)",
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.info("Density curve not generated for this issue.")

                    # Metrics badge block
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Predicted Mean Return", format_pct(pred.get("mean_pct")))
                    m_col2.metric("Probability of Listing Loss", f"{pred.get('p_loss', 0)*100:.1f}%")
                    m_col3.metric("80% Credible Interval", f"[{pred.get('q10_pct', 0):.1f}% → {pred.get('q90_pct', 0):.1f}%]")

                with col_right:
                    st.markdown("#### 📊 Subscription & GMP Velocity")

                    # Subscription bar chart
                    sub = detail.get("subscription") or {}
                    cats = ["QIB", "sNII", "bNII", "Retail (RII)", "Total"]
                    vals = [sub.get("qib"), sub.get("snii"), sub.get("bnii"), sub.get("rii"), sub.get("total")]
                    df_sub = pd.DataFrame({"Category": cats, "Subscription (x)": [v if v is not None else 0 for v in vals]})

                    fig_sub = px.bar(
                        df_sub,
                        x="Category",
                        y="Subscription (x)",
                        color="Category",
                        title="Live Category Oversubscription Levels",
                        color_discrete_sequence=px.colors.qualitative.Prism,
                    )
                    fig_sub.add_hline(y=1.0, line_dash="dash", line_color="#f59e0b", annotation_text="1.0x (Fully Subscribed)")
                    fig_sub.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.6)",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_sub, use_container_width=True)

                    # GMP History line chart
                    gmp_hist = detail.get("gmp_history") or []
                    if gmp_hist:
                        df_gmp = pd.DataFrame(gmp_hist)
                        fig_gmp = px.line(
                            df_gmp,
                            x="date",
                            y="gmp_rs",
                            title="Grey Market Premium (GMP ₹) Trend",
                            markers=True,
                            color_discrete_sequence=["#10b981"],
                        )
                        fig_gmp.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15, 23, 42, 0.6)",
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        st.plotly_chart(fig_gmp, use_container_width=True)
                    else:
                        st.caption("No historical GMP track recorded yet for this issue.")

            with tab_whatif:
                st.markdown("#### 🎛️ Real-time What-If Sensitivity Simulator")
                st.markdown(
                    """
                    Adjust hypothetical final market parameters (e.g. if QIB subscription explodes on Day 3 or GMP falls)
                    and see the model re-compute the entire listing return distribution and expected profit in real time!
                    """
                )

                sub = detail.get("subscription") or {}
                current_gmp_pct = float(detail.get("gmp_consensus_pct") or 0.0)
                current_qib = float(sub.get("qib") or 1.0)
                current_nii = float(sub.get("nii") or 1.0)
                current_rii = float(sub.get("rii") or 1.0)

                w_c1, w_c2 = st.columns(2)
                with w_c1:
                    sim_gmp_pct = st.slider("Hypothetical GMP (%)", min_value=-30.0, max_value=200.0, value=current_gmp_pct, step=1.0)
                    sim_qib = st.slider("Hypothetical QIB Subscription (x)", min_value=0.5, max_value=300.0, value=max(1.0, current_qib), step=1.0)
                with w_c2:
                    sim_nii = st.slider("Hypothetical NII Subscription (x)", min_value=0.5, max_value=250.0, value=max(1.0, current_nii), step=1.0)
                    sim_rii = st.slider("Hypothetical Retail Subscription (x)", min_value=0.5, max_value=150.0, value=max(1.0, current_rii), step=0.5)

                if st.button("🚀 Re-run Deep MDN Model with Overrides"):
                    with st.spinner("Computing Monte-Carlo joint scenarios..."):
                        overrides = {
                            "gmp_pct": sim_gmp_pct,
                            "sub_qib": sim_qib,
                            "sub_nii": sim_nii,
                            "sub_rii": sim_rii,
                            "final_subscription": True,
                        }
                        sim_detail = service.ipo_detail(selected_slug, overrides)
                        if sim_detail:
                            s_pred = sim_detail.get("prediction") or {}
                            s_ev = sim_detail.get("min_application_ev") or {}

                            st.success("Simulation complete!")
                            r1, r2, r3, r4 = st.columns(4)
                            r1.metric("Simulated Return", format_pct(s_pred.get("mean_pct")), delta=f"{(s_pred.get('mean_pct', 0) - pred.get('mean_pct', 0)):.1f}%")
                            r2.metric("P(Loss)", f"{s_pred.get('p_loss', 0)*100:.1f}%", delta=f"{(s_pred.get('p_loss', 0) - pred.get('p_loss', 0))*100:.1f}%", delta_color="inverse")
                            r3.metric("Retail Allotment Odds", f"{sim_detail.get('allotment', {}).get('p_mean', 0)*100:.1f}%")
                            r4.metric("Simulated Profit / Lot", format_inr(s_ev.get("ev_rupees")))

            with tab_data:
                st.markdown("#### 📰 Recent News & Sentiment")
                news = detail.get("news") or []
                if news:
                    for n in news[:8]:
                        st.markdown(f"- **[{n.get('title')}]({n.get('url')})** — *{n.get('publisher') or 'Financial News'}*")
                else:
                    st.caption("No recent headlines captured.")


# ===========================================================================
# PAGE 3: 🎯 CAPITAL OPTIMIZER PLANNER
# ===========================================================================
elif page == "🎯 Capital Optimizer Planner":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">🎯 Family Capital Allocation Optimizer</h1>
            <p style="margin: 5px 0 0; color: #94a3b8; font-size: 0.95rem;">
                Solves the multi-PAN integer programming optimization problem (Google OR-Tools CP-SAT) across active IPOs,
                enforcing SEBI one-application-per-PAN rules, ASBA fund blocking dates, and joint copula portfolio returns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_setup, col_result = st.columns([1, 2])

    with col_setup:
        st.markdown("### 💼 Portfolio Configuration")
        total_capital = st.number_input("Total Family Capital Available (₹)", min_value=15000, max_value=10000000, value=200000, step=15000)

        num_applicants = st.slider("Number of Family PAN Applicants", min_value=1, max_value=8, value=3)
        applicants = []
        for i in range(num_applicants):
            with st.expander(f"Applicant {i+1} Details", expanded=(i == 0)):
                a_name = st.text_input(f"Name #{i+1}", value=f"Family Member {i+1}", key=f"name_{i}")
                a_pan = st.text_input(f"Unique PAN / Identifier #{i+1}", value=f"PAN{i+1}", key=f"pan_{i}")
                a_nii = st.checkbox(f"Allow Small NII (₹2-10L)", value=True, key=f"nii_{i}")
                applicants.append({"name": a_name, "pan": a_pan, "allow_nii": a_nii})

        opt_objective = st.selectbox(
            "Optimization Strategy",
            [
                ("compare", "⚖️ Compare All 4 Objectives"),
                ("max_ev", "🏆 Maximize Expected Profit"),
                ("max_allot_prob", "🎯 Maximize Probability of ≥1 Allotment"),
                ("capital_efficiency", "⚡ Maximize Return on Blocked Capital"),
                ("risk_adjusted", "🛡️ Downside Risk Adjusted (CVaR)"),
            ],
            format_func=lambda x: x[1],
        )[0]

        risk_appetite = st.select_slider("Risk Tolerance", ["conservative", "moderate", "aggressive"], value="moderate")
        allow_sme = st.checkbox("Include SME IPOs", value=True)
        boards = ["mainboard", "sme"] if allow_sme else ["mainboard"]

        btn_plan = st.button("🚀 Optimize Portfolio Allocation", type="primary")

    with col_result:
        st.markdown("### 📊 Optimal Allocation Solution")

        if btn_plan:
            with st.spinner("Solving Integer Programming Knapsack & Monte-Carlo Simulation..."):
                payload = {
                    "capital": total_capital,
                    "applicants": applicants,
                    "objective": opt_objective,
                    "risk": risk_appetite,
                    "boards": boards,
                    "include_demo": include_demo,
                }
                res = service.make_plan(payload)

                if opt_objective == "compare":
                    plans = res.get("plans") or {}
                    if not plans:
                        st.warning("No candidate IPOs were eligible for bidding.")
                    else:
                        st.markdown("#### ⚖️ Comparison of Strategies")
                        comp_data = []
                        for k, p in plans.items():
                            s = p.get("sim") or {}
                            comp_data.append({
                                "Strategy": p.get("label"),
                                "Expected Profit": format_inr(s.get("mean_profit")),
                                "Return on Capital": format_pct(s.get("mean_return_pct")),
                                "P(≥1 Allotment)": f"{s.get('p_at_least_one', 0)*100:.1f}%",
                                "Capital Blocked": format_inr(p.get("max_blocked")),
                                "Applications": len(p.get("applications") or []),
                            })
                        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

                        # Detailed breakdown of max_ev
                        best = plans.get("max_ev") or next(iter(plans.values()))
                        st.markdown(f"#### 📝 Allocation Schedule ({best.get('label')})")
                        apps = best.get("applications") or []
                        if apps:
                            app_df = pd.DataFrame([{
                                "Applicant": a.get("applicant_name"),
                                "IPO": a.get("ipo_name"),
                                "Category": a.get("category"),
                                "Lots": a.get("lots"),
                                "Blocked (₹)": format_inr(a.get("amount")),
                            } for a in apps])
                            st.dataframe(app_df, hide_index=True, use_container_width=True)
                        else:
                            st.info("No applications allocated under current constraints.")
                else:
                    # Single plan output
                    sim = res.get("sim") or {}
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Expected Portfolio Profit", format_inr(sim.get("mean_profit")))
                    s2.metric("P(≥1 Allotment)", f"{sim.get('p_at_least_one', 0)*100:.1f}%")
                    s3.metric("Peak Capital Blocked", format_inr(res.get("max_blocked")))

                    apps = res.get("applications") or []
                    if apps:
                        st.markdown("#### 📝 Recommended Applications")
                        app_df = pd.DataFrame([{
                            "Applicant": a.get("applicant_name"),
                            "IPO": a.get("ipo_name"),
                            "Category": a.get("category"),
                            "Lots": a.get("lots"),
                            "Blocked (₹)": format_inr(a.get("amount")),
                        } for a in apps])
                        st.dataframe(app_df, hide_index=True, use_container_width=True)
        else:
            st.info("Configure your family capital and applicants on the left, then click **'Optimize Portfolio Allocation'**.")


# ===========================================================================
# PAGE 4: 🧮 ALLOTMENT ODDS CALCULATOR
# ===========================================================================
elif page == "🧮 Allotment Odds Calculator":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">🧮 Standalone Allotment Probability Calculator</h1>
            <p style="margin: 5px 0 0; color: #94a3b8; font-size: 0.95rem;">
                Computes exact SEBI ICDR lottery odds for Retail (RII) and NII categories, modelling the uncertain average application size $L$
                and the correlation of multiple family PAN applications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1, 1])

    with c_left:
        calc_board = st.selectbox("IPO Board", ["mainboard", "sme"], format_func=lambda x: x.upper())
        calc_cat = st.selectbox("Category", ["RII", "SNII", "BNII"], format_func=lambda x: "Retail Individual (RII)" if x == "RII" else ("Small NII (₹2-10L)" if x == "SNII" else "Big NII (>₹10L)"))
        calc_sub = st.slider("Category Subscription Times (x)", min_value=1.0, max_value=150.0, value=25.0, step=0.5)
        calc_apps = st.slider("Number of Family PAN Applications", min_value=1, max_value=10, value=3)

        rng = np.random.default_rng(42)
        n_samples = 20000
        L = sample_L(calc_board, calc_cat, n_samples, rng)
        p_samples, _ = allot_given(np.full(n_samples, calc_sub), L, 1)

        single_p = float(p_samples.mean())
        joint_p = prob_at_least_one(p_samples, calc_apps)
        naive_p = naive_at_least_one(single_p, calc_apps)

        st.markdown("---")
        st.markdown("### 📊 Computed Probability Metrics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Single Application Odds", f"{single_p*100:.2f}%")
        k2.metric(f"Odds across {calc_apps} Family PANs", f"{joint_p*100:.2f}%")
        k3.metric("Naive Independent Odds", f"{naive_p*100:.2f}%", help="Why the difference? Because all family members share the same unknown demand shock.")

    with c_right:
        st.markdown("### 📈 Odds Scaling Curve Across Family Members")
        # Generate curve for 1 to 10 applicants
        table_rows = []
        for k in range(1, 11):
            table_rows.append({
                "Family PANs": k,
                "True Allotment Odds (%)": prob_at_least_one(p_samples, k) * 100,
                "Naive Coin-toss Odds (%)": naive_at_least_one(single_p, k) * 100,
            })
        df_curve = pd.DataFrame(table_rows)

        fig_curve = px.line(
            df_curve,
            x="Family PANs",
            y=["True Allotment Odds (%)", "Naive Coin-toss Odds (%)"],
            markers=True,
            title=f"Allotment Probability vs. Family Applications at {calc_sub}x Subscription",
            color_discrete_sequence=["#3b82f6", "#94a3b8"],
        )
        fig_curve.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        st.markdown(
            """
            <div class="callout-box">
                <b>SEBI Retail Lottery Rule:</b><br/>
                Regardless of whether an investor bids for 1 lot or maximum retail lots, if oversubscribed, they enter the lottery for exactly ONE lot.
                Therefore, applying with multiple family PANs (each 1 lot) is mathematically superior to bidding multiple lots on a single PAN.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 5: 📜 HISTORICAL DATA & BACKTEST
# ===========================================================================
elif page == "📜 Historical Data & Backtest":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">📜 Historical IPO Database & Backtest Performance</h1>
            <p style="margin: 5px 0 0; color: #94a3b8; font-size: 0.95rem;">
                Verified dataset of 2,200+ listed Indian IPOs since 2012 with real listing gains, board comparisons, and walk-forward strategy backtests.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_hist = get_history_ipos()

    if df_hist.empty:
        st.info("No historical records found in local SQLite database. Ensure backend data ingestion is initialized.")
    else:
        # Filters
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            hist_board = st.selectbox("Filter Board", ["All", "mainboard", "sme"])
        with h_col2:
            years = sorted(df_hist["year"].dropna().unique().astype(int), reverse=True)
            hist_year = st.selectbox("Filter Year", ["All Years"] + list(years))
        with h_col3:
            search_query = st.text_input("Search IPO Name or Symbol", "")

        df_filtered = df_hist.copy()
        if hist_board != "All":
            df_filtered = df_filtered[df_filtered["board"] == hist_board]
        if hist_year != "All Years":
            df_filtered = df_filtered[df_filtered["year"] == int(hist_year)]
        if search_query:
            df_filtered = df_filtered[
                df_filtered["name"].str.contains(search_query, case=False, na=False)
                | df_filtered["symbol"].str.contains(search_query, case=False, na=False)
            ]

        # Key stats
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        s_c1.metric("Total Issues", len(df_filtered))
        s_c2.metric("Average Listing Gain", format_pct(df_filtered["listing_gain_pct"].mean()))
        s_c3.metric("Median Listing Gain", format_pct(df_filtered["listing_gain_pct"].median()))
        neg_rate = (df_filtered["listing_gain_pct"] < 0).mean() * 100 if len(df_filtered) else 0
        s_c4.metric("Loss Rate (% Negative)", f"{neg_rate:.1f}%")

        # Visualizations
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_hist_dist = px.histogram(
                df_filtered,
                x="listing_gain_pct",
                nbins=50,
                title="Listing Day Return Distribution (%)",
                color_discrete_sequence=["#3b82f6"],
                labels={"listing_gain_pct": "Listing Gain (%)"},
            )
            fig_hist_dist.add_vline(x=0, line_dash="dash", line_color="#ef4444")
            fig_hist_dist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist_dist, use_container_width=True)

        with ch2:
            annual = df_hist.groupby(["year", "board"])["listing_gain_pct"].mean().reset_index()
            fig_annual = px.bar(
                annual,
                x="year",
                y="listing_gain_pct",
                color="board",
                barmode="group",
                title="Average Listing Gain by Year & Board",
                labels={"listing_gain_pct": "Mean Gain (%)", "year": "Year"},
                color_discrete_map={"mainboard": "#3b82f6", "sme": "#c084fc"},
            )
            fig_annual.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_annual, use_container_width=True)

        st.dataframe(
            df_filtered[["name", "board", "listing_date", "issue_price", "issue_size_cr", "listing_gain_pct", "sector"]],
            use_container_width=True,
            hide_index=True,
        )


# ===========================================================================
# PAGE 6: 🧠 AI MODEL ARCHITECTURE
# ===========================================================================
elif page == "🧠 AI Model Architecture":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">🧠 AI Deep Learning Model Card & Architecture</h1>
            <p style="margin: 5px 0 0; color: #94a3b8; font-size: 0.95rem;">
                Mathematical formulation, walk-forward out-of-time evaluation, and Deep MDN ensemble specifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("### 🧬 Architecture Overview")
        st.markdown(
            """
            - **Input Modalities**:
              - *Tabular Fundamentals*: Issue size, price band ratio, anchor allocation %, sector, PE ratio.
              - *Time-Series Dynamics*: QIB, NII, Retail subscription velocity trajectories.
              - *Grey Market Premium (GMP)*: Observation history and consensus spreads.
              - *Financial News Sentiment*: FinBERT sentiment scores on headlines.
            - **Core Network**:
              - Deep ResNet-style MLP with Highway blocks, LayerNorm, and Dropout.
            - **Output Layer**:
              - **Mixture Density Network (MDN)** predicting Gaussian mixture parameters: $(\pi_k, \mu_k, \sigma_k)_{k=1}^K$.
              - Captures bimodal distributions (e.g. either huge listing pop or discount flop).
            - **Ensemble**:
              - 5 walk-forward snapshot ensemble members trained across distinct temporal folds.
            """
        )

    with m_col2:
        st.markdown("### 🧪 Model Metrics & Benchmarks")
        metrics_file = settings.models_dir / "metrics.json"
        if metrics_file.exists():
            try:
                m_data = json.loads(metrics_file.read_text())
                st.json({
                    "Model Version": m_data.get("version"),
                    "Ensemble Size": m_data.get("ensemble_size"),
                    "Training Issues": m_data.get("rows", {}).get("train"),
                    "Test Issues": m_data.get("rows", {}).get("test"),
                    "Validation Split Date": m_data.get("split", {}).get("valid"),
                })
            except Exception:
                st.caption("Metrics loaded.")
        else:
            st.info("Metrics card not generated yet.")

        backtest_file = settings.models_dir / "backtest.json"
        if backtest_file.exists():
            st.markdown("### 🏆 Strategy Backtest Summary")
            try:
                b_data = json.loads(backtest_file.read_text())
                summary_df = pd.DataFrame(b_data.get("summary", []))
                if not summary_df.empty:
                    st.dataframe(
                        summary_df[["strategy", "total_expected_profit", "sharpe_like", "loss_window_share", "avg_capital_deployed"]],
                        hide_index=True,
                        use_container_width=True,
                    )
            except Exception as e:
                st.caption(f"Backtest details: {e}")
