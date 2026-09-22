/**
 * App shell — sidebar + routed main area, matching predictions.py's
 * ``st.set_page_config(layout="wide", initial_sidebar_state="expanded")``.
 */
import { useEffect, useMemo, useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import { PageTitleContext, SidebarExtraContext } from "./sidebarContext";
import { useApi, useThemeOverride } from "./useApi";

import Home from "./pages/Home";
import WeeklyPredictions from "./pages/WeeklyPredictions";
import ValueBets from "./pages/ValueBets";
import TeamExplorer from "./pages/TeamExplorer";
import HistoricalAnalysis from "./pages/HistoricalAnalysis";
import ModelPerformance from "./pages/ModelPerformance";
import WinProbability from "./pages/WinProbability";
import PreseasonOutlook from "./pages/PreseasonOutlook";
import DataQuality from "./pages/DataQuality";
import TotalMarketSignals from "./pages/TotalMarketSignals";

function resolveTheme(override) {
  if (override === "day" || override === "night") return override;
  const hour = new Date().getHours();
  return hour >= 6 && hour < 22 ? "day" : "night";
}

export default function App() {
  const { data: meta } = useApi("/meta", {});
  const [extra, setExtra] = useState(null);
  const override = useThemeOverride();
  const location = useLocation();

  // Resolve the current route's <h1> from /api/meta so the shell can paint
  // before the page payload arrives.
  const pageTitle = useMemo(() => {
    if (!meta) return null;
    const routes = [
      ...meta.nav.flatMap((section) => section.pages),
      ...(meta.extra_routes || []),
    ];
    const match =
      routes.find((page) => page.path === location.pathname) ||
      routes.find((page) => page.path === "/");
    return match?.page_title ?? null;
  }, [meta, location.pathname]);

  useEffect(() => {
    document.documentElement.dataset.theme = resolveTheme(override);
  }, [override]);

  return (
    <SidebarExtraContext.Provider value={setExtra}>
      <PageTitleContext.Provider value={pageTitle}>
        <div className="app-shell">
          {/* predictions.py::home_page calls render_sidebar(show_logo=False). */}
          <Sidebar meta={meta} extra={extra} showLogo={location.pathname !== "/"} />
          <main className="main" data-testid="stAppViewContainer">
            <div className="block-container" data-testid="stMainBlockContainer">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/Weekly_Predictions" element={<WeeklyPredictions />} />
                <Route path="/Value_Bets" element={<ValueBets />} />
                <Route path="/Team_Explorer" element={<TeamExplorer />} />
                <Route path="/Historical_Analysis" element={<HistoricalAnalysis />} />
                <Route path="/Model_Performance" element={<ModelPerformance />} />
                <Route path="/Win_Probability" element={<WinProbability />} />
                <Route path="/Preseason_Outlook" element={<PreseasonOutlook />} />
                <Route path="/Data_Quality" element={<DataQuality />} />
                <Route path="/Total_Market_Signals" element={<TotalMarketSignals />} />
                <Route path="*" element={<Home />} />
              </Routes>
            </div>
          </main>
        </div>
      </PageTitleContext.Provider>
    </SidebarExtraContext.Provider>
  );
}
