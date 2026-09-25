/** Preseason Outlook — mirrors ``pages/8_Preseason_Outlook.py``.
 *
 * The Streamlit page adds its own "Filters" block to the sidebar, so this
 * component publishes the season selector through the sidebar-extra channel.
 */
import { useEffect, useState } from "react";
import PageState from "../components/PageState";
import { PlotlyChart } from "../components/LazyPlotlyChart";
import {
  Alert,
  Caption,
  Columns,
  DataFrame,
  Divider,
  Heading,
  Markdown,
  SelectBox,
  Tabs,
} from "../components/ui";
import { useApi } from "../useApi";
import { useSidebarExtra } from "../sidebarContext";

export default function PreseasonOutlook() {
  const [season, setSeason] = useState(null);
  const [conference, setConference] = useState("All");
  const setSidebarExtra = useSidebarExtra();

  const state = useApi("/preseason-outlook", { season, conference });
  const sidebar = state.data?.sidebar;

  useEffect(() => {
    if (sidebar && season === null) setSeason(sidebar.season);
  }, [sidebar, season]);

  useEffect(() => {
    if (!sidebar) return undefined;
    setSidebarExtra(
      <div data-testid="preseasonSidebarFilters">
        <Divider />
        <Heading level={2} className="sidebar-heading">
          Filters
        </Heading>
        <SelectBox
          label="Season"
          options={sidebar.seasons}
          value={sidebar.season}
          onChange={(value) => setSeason(Number(value))}
        />
      </div>
    );
    return () => setSidebarExtra(null);
  }, [sidebar, setSidebarExtra]);

  return (
    <PageState state={state}>
      {(data) => (
        <>
          <Heading level={1}>{data.title}</Heading>
          <Caption>{data.caption}</Caption>

          <Tabs
            tabs={data.tabs}
            renderPanel={(tab) => {
              if (tab.id === "returning") {
                return (
                  <>
                    <Heading level={3}>{tab.heading}</Heading>
                    {tab.info ? <Alert kind="info">{tab.info}</Alert> : null}
                    {tab.warning ? <Alert kind="warning">{tab.warning}</Alert> : null}
                    {tab.table ? (
                      <>
                        <DataFrame table={tab.table} height={tab.table.height} />
                        {tab.figure ? <PlotlyChart figure={tab.figure} /> : null}
                      </>
                    ) : null}
                  </>
                );
              }
              if (tab.id === "portal") {
                return (
                  <>
                    <Heading level={3}>{tab.heading}</Heading>
                    {tab.info ? <Alert kind="info">{tab.info}</Alert> : null}
                    {tab.warning ? <Alert kind="warning">{tab.warning}</Alert> : null}
                    {tab.gainers ? (
                      <>
                        <Columns>
                          <div>
                            <Markdown>{tab.gainers_heading}</Markdown>
                            <DataFrame table={tab.gainers} />
                          </div>
                          <div>
                            <Markdown>{tab.losers_heading}</Markdown>
                            <DataFrame table={tab.losers} />
                          </div>
                        </Columns>
                        <PlotlyChart figure={tab.figure} />
                      </>
                    ) : null}
                  </>
                );
              }
              return (
                <>
                  <Heading level={3}>{tab.heading}</Heading>
                  <Caption>{tab.caption}</Caption>
                  {tab.info ? <Alert kind="info">{tab.info}</Alert> : null}
                  {tab.warning ? <Alert kind="warning">{tab.warning}</Alert> : null}
                  {tab.conferences ? (
                    <SelectBox
                      label="Conference"
                      options={tab.conferences}
                      value={tab.conference}
                      onChange={(value) => setConference(value)}
                    />
                  ) : null}
                  {tab.figure ? <PlotlyChart figure={tab.figure} /> : null}
                </>
              );
            }}
          />
        </>
      )}
    </PageState>
  );
}
