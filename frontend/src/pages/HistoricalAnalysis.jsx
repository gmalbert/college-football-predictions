/** Historical Analysis — mirrors ``pages/4_Historical_Analysis.py``. */
import { useEffect, useMemo, useState } from "react";
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import { PlotlyChart } from "../components/LazyPlotlyChart";
import {
  Alert,
  Columns,
  DataFrame,
  Heading,
  MetricRow,
  SelectBox,
  Tabs,
} from "../components/ui";
import { useApi } from "../useApi";

export default function HistoricalAnalysis() {
  const [seasonFrom, setSeasonFrom] = useState(null);
  const [seasonTo, setSeasonTo] = useState(null);
  const [teamA, setTeamA] = useState(null);
  const [teamB, setTeamB] = useState(null);

  const params = useMemo(
    () => ({ season_from: seasonFrom, season_to: seasonTo, team_a: teamA, team_b: teamB }),
    [seasonFrom, seasonTo, teamA, teamB]
  );
  const state = useApi("/historical-analysis", params);

  useEffect(() => {
    const data = state.data;
    if (!data?.seasons) return;
    if (seasonFrom === null) setSeasonFrom(data.selection.season_from);
    if (seasonTo === null) setSeasonTo(data.selection.season_to);
    if (teamA === null) setTeamA(data.selection.team_a);
    if (teamB === null) setTeamB(data.selection.team_b);
  }, [state.data, seasonFrom, seasonTo, teamA, teamB]);

  return (
    <PageState state={state}>
      {(data) => (
        <>
          <Heading level={1}>{data.title}</Heading>
          {(data.warnings || []).map((text) => (
            <Alert key={text} kind="warning">
              {text}
            </Alert>
          ))}

          {data.stopped ? null : (
            <>
              <Columns>
                <SelectBox
                  label="Season From"
                  options={data.seasons}
                  value={data.selection.season_from}
                  onChange={(value) => setSeasonFrom(Number(value))}
                />
                <SelectBox
                  label="Season To"
                  options={data.seasons}
                  value={data.selection.season_to}
                  onChange={(value) => setSeasonTo(Number(value))}
                />
              </Columns>

              <Tabs
                tabs={data.tabs}
                renderPanel={(tab) => {
                  if (tab.id === "trends") {
                    return (
                      <>
                        <Heading level={3}>{tab.heading}</Heading>
                        <PlotlyChart figure={tab.score_figure} />
                        <PlotlyChart figure={tab.hfa_figure} />
                        {tab.ats_table ? (
                          <>
                            <Heading level={3}>{tab.ats_table.heading}</Heading>
                            <DataFrame table={tab.ats_table} />
                          </>
                        ) : null}
                      </>
                    );
                  }
                  if (tab.id === "h2h") {
                    return (
                      <>
                        <Heading level={3}>{tab.heading}</Heading>
                        <Columns>
                          <SelectBox
                            label="Team A"
                            options={data.teams}
                            value={data.selection.team_a}
                            onChange={(value) => setTeamA(value)}
                          />
                          <SelectBox
                            label="Team B"
                            options={data.teams}
                            value={data.selection.team_b}
                            onChange={(value) => setTeamB(value)}
                          />
                        </Columns>
                        {tab.info ? (
                          <Alert kind="info">{tab.info}</Alert>
                        ) : (
                          <>
                            <MetricRow metrics={tab.metrics} />
                            <DataFrame table={tab.table} />
                          </>
                        )}
                      </>
                    );
                  }
                  return (
                    <>
                      <Heading level={3}>{tab.heading}</Heading>
                      {tab.info ? (
                        <Alert kind="info">{tab.info}</Alert>
                      ) : (
                        <>
                          <PlotlyChart figure={tab.figure} />
                          <Heading level={3}>{tab.table.heading}</Heading>
                          <DataFrame table={tab.table} />
                        </>
                      )}
                    </>
                  );
                }}
              />
            </>
          )}
          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
