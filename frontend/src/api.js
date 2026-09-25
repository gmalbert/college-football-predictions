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

// A cold instance has to load its artifacts before it can answer, so the
// ceiling has to clear a cold start rather than a warm request. Without a
// deadline a slow or black-holed request leaves the page on "Loading…" forever,
// which is indistinguishable from a hang.
const REQUEST_TIMEOUT_MS = 60_000;

export async function apiGet(path, params, signal) {
  const deadline = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const combined = signal ? AbortSignal.any([signal, deadline]) : deadline;

  let response;
  try {
    response = await fetch(buildUrl(path, params), { signal: combined });
  } catch (error) {
    // A caller-requested abort (unmount, param change) stays an AbortError and
    // is ignored upstream; only a passed deadline becomes a visible message.
    if (deadline.aborted) {
      throw new Error(
        `The server did not respond within ${REQUEST_TIMEOUT_MS / 1000}s. ` +
          "A free instance can take a moment to wake up — try again.",
      );
    }
    throw error;
  }

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
