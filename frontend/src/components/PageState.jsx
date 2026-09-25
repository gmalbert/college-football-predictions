/** Shared loading / error boundary for a page's API payload. */
import { usePageTitle } from "../sidebarContext";

export function PageState({ state, children }) {
  const title = usePageTitle();

  if (state.error) {
    return <div className="page-error">Failed to load page data: {state.error}</div>;
  }

  if (state.loading && !state.data) {
    // Paint the heading straight away.  The title comes from /api/meta (a few
    // hundred bytes) rather than the page payload, so the page is readable
    // while the data request is still in flight — the same progressive render
    // Streamlit does.  The real <h1> replaces this one with identical text.
    return (
      <div data-testid="pageShell">
        {title ? (
          <h1 className="stHeading" data-testid="stHeading">
            {title}
          </h1>
        ) : null}
        <div className="loading">Loading…</div>
      </div>
    );
  }

  if (!state.data) return null;
  return children(state.data);
}

export default PageState;
