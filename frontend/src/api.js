/**
 * Thin fetch layer over the FastAPI backend.
 *
 * The client sends its IANA timezone on the Weekly Predictions request so the
 * kickoff column matches Streamlit's ``st.context.timezone`` behaviour.
 */

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export function browserTimezone() {
  // ``?tz=`` lets the parity harness pin the timezone so the React build can be
  // compared against a headless Streamlit run (which reports UTC).
  try {
    const override = new URLSearchParams(window.location.search).get("tz");
    if (override) return override;
  } catch {
    /* ignore */
  }
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

function buildUrl(path, params) {
  const query = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    query.append(key, String(value));
  });
  const suffix = query.toString();
  return `${API_BASE}${path}${suffix ? `?${suffix}` : ""}`;
}

export async function apiGet(path, params, signal) {
  const response = await fetch(buildUrl(path, params), { signal });
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText} ${text}`.trim());
  }
  return response.json();
}

export function weeklyParams(state) {
  return {
    season: state.season,
    week: state.week,
    conference: state.conference,
    min_edge: state.min_edge,
    sort_by: state.sort_by,
    tz: browserTimezone(),
  };
}
