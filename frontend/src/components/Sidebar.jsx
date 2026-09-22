/**
 * Sidebar — mirrors Streamlit's sidebar composition.
 *
 * Streamlit renders the page navigation *above* whatever the page adds with
 * ``st.sidebar`` (``stSidebarNav`` then ``stSidebarUserContent``).  The page
 * content is the logo, divider, live metrics, plus anything a page injects
 * (Preseason Outlook adds a "Filters" block).
 *
 * ``showLogo`` mirrors ``render_sidebar(show_logo=...)`` — the home page passes
 * False, every other page leaves it at the default.
 */
import { NavLink } from "react-router-dom";
import { Divider, Metric } from "./ui";

export function Sidebar({ meta, extra, showLogo = true }) {
  if (!meta) {
    return <aside className="sidebar" data-testid="stSidebar" />;
  }
  const { sidebar, nav } = meta;

  return (
    <aside className="sidebar" data-testid="stSidebar">
      <nav className="sidebar-nav" data-testid="stSidebarNav">
        {nav.map((section, index) => (
          <div key={section.section || `section-${index}`}>
            {section.section ? (
              <div className="nav-section-label" data-testid="stNavSectionHeader">
                {section.section}
              </div>
            ) : null}
            <div data-testid="stSidebarNavItems">
              {section.pages.map((page) => (
                <NavLink
                  key={page.path}
                  to={page.path}
                  end={page.path === "/"}
                  className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
                  data-testid="stSidebarNavLink"
                >
                  <span className="nav-icon">{page.icon}</span> {page.title}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="sidebar-user-content" data-testid="stSidebarUserContent">
        {showLogo && sidebar?.logo ? (
          <img className="sidebar-logo" src={sidebar.logo} alt="Tailgate Edge" />
        ) : null}

        <Divider />

        {(sidebar?.metrics || []).map((metric) => (
          <Metric key={metric.label} {...metric} />
        ))}

        {extra}
      </div>
    </aside>
  );
}

export default Sidebar;
