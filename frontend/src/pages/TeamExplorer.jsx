/** Team Explorer — mirrors ``pages/3_Team_Explorer.py``. */
import { useEffect, useMemo, useState } from "react";
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import { PlotlyChart } from "../components/LazyPlotlyChart";
import {
  Alert,
  Caption,
  Columns,
  DataFrame,
  Divider,
  Heading,
  MetricRow,
  SelectBox,
} from "../components/ui";
import { useApi } from "../useApi";

export default function TeamExplorer() {
  const [team, setTeam] = useState(null);
  const [season, setSeason] = useState(null);

  const params = useMemo(() => ({ team, season }), [team, season]);
  const state = useApi("/team-explorer", params);

  useEffect(() => {
    if (!state.data?.teams) return;
    if (team === null) setTeam(state.data.selection.team);
    if (season === null) setSeason(state.data.selection.season);
  }, [state.data, team, season]);

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
              <Columns widths={[3, 1]}>
                <SelectBox
                  label="Team"
                  options={data.teams}
                  value={data.selection.team}
                  onChange={(value) => setTeam(value)}
                />
                <SelectBox
                  label="Season"
                  options={data.seasons}
                  value={data.selection.season}
                  onChange={(value) => setSeason(Number(value))}
                />
              </Columns>

              <Heading level={3}>{data.heading}</Heading>
              <MetricRow metrics={data.metrics} />
              <Divider />

              {data.elo.figure ? <PlotlyChart figure={data.elo.figure} /> : null}
              {data.elo.info ? <Alert kind="info">{data.elo.info}</Alert> : null}
              <Divider />

              {data.radar.figure ? <PlotlyChart figure={data.radar.figure} /> : null}
              <Divider />

              {data.quadrant.figure ? (
                <>
                  <Heading level={3}>{data.quadrant.heading}</Heading>
                  <Caption>{data.quadrant.caption}</Caption>
                  <PlotlyChart figure={data.quadrant.figure} />
                </>
              ) : null}
              <Divider />

              <Heading level={3}>{data.schedule.heading}</Heading>
              {data.schedule.info ? (
                <Alert kind="info">{data.schedule.info}</Alert>
              ) : (
                <DataFrame table={data.schedule} />
              )}
            </>
          )}
          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
