"""Data quality, leakage, freshness, and model release checks."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from footer import add_betting_oracle_footer
from utils.repo_audit import run_repository_audit
from utils.release import load_current_release
from utils.ui_components import render_sidebar, themed_dataframe


render_sidebar()
st.title("🛡️ Data & Model Quality")
st.caption("Release-contract checks for grain, point-in-time safety, freshness, and evaluation scope.")


@st.cache_data(ttl=900)
def _audit() -> dict:
    return run_repository_audit()


report = _audit()
release = load_current_release()
summary = report["summary"]
col1, col2, col3 = st.columns(3)
col1.metric("Passing", summary["pass"])
col2.metric("Warnings", summary["warn"])
col3.metric("Failures", summary["fail"])

if release:
    st.caption(
        f"Release {release.get('release_id', 'unknown')[:12]} · "
        f"{release.get('status', 'hold').upper()} · "
        f"generated {release.get('generated_at', 'unknown')}"
    )
else:
    st.warning("No release metadata is published. Treat generated artifacts as unreleased.")

checks = pd.DataFrame(report["checks"])
if not checks.empty:
    checks["Status"] = checks["status"].map(
        {"pass": "✅ Pass", "warn": "⚠️ Warning", "fail": "❌ Fail"}
    )
    checks["value"] = checks["value"].map(
        lambda value: "—" if value is None else str(value)
    )
    themed_dataframe(
        checks[["Status", "name", "message", "value"]].rename(
            columns={"name": "Check", "message": "Details", "value": "Value"}
        ),
        width="stretch",
        hide_index=True,
    )

st.info(
    "A warning does not automatically block the dashboard. A failed grain, "
    "point-in-time, or out-of-sample evaluation check should block model promotion."
)
add_betting_oracle_footer()
