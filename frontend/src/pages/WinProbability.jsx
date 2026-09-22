/** Win Probability — mirrors ``pages/7_Win_Probability.py``. */
import { useEffect, useMemo, useState } from "react";
import PageState from "../components/PageState";
import { PlotlyChart } from "../components/LazyPlotlyChart";
import {
  Alert,
  Caption,
  Columns,
  DataFrame,
  Divider,
  Expander,
  Heading,
  MetricRow,
  SelectBox,
  TextInput,
} from "../components/ui";
import { useApi } from "../useApi";

export default function WinProbability() {
  const [season, setSeason] = useState(2025);
  const [seasonType, setSeasonType] = useState("Regular");
  const [week, setWeek] = useState(1);
  const [search, setSearch] = useState("");
  const [gameId, setGameId] = useState(null);

  const params = useMemo(
    () => ({
      season,
      season_type: seasonType,
      week,
      search,
      game_id: gameId ?? undefined,
    }),
    [season, seasonType, week, search, gameId]
  );
  const state = useApi("/win-probability", params);
  const controls = state.data?.controls;

  useEffect(() => {
    if (controls && gameId === null) setGameId(controls.game_id);
  }, [controls, gameId]);

  return (
    <PageState state={state}>
      {(data) => (
        <>
          <Heading level={1}>{data.title}</Heading>
          <Caption>{data.caption}</Caption>

          {(data.warnings || []).map((text) => (
            <Alert key={text} kind="warning">
              {text}
            </Alert>
          ))}
          {data.info ? <Alert kind="info">{data.info}</Alert> : null}

          <Columns widths={[2, 2, 3]}>
            <SelectBox
              label="Season"
              options={data.controls.seasons}
              value={data.controls.season}
              onChange={(value) => {
                setSeason(Number(value));
                setGameId(null);
              }}
            />
            <SelectBox
              label="Season type"
              options={data.controls.season_types}
              value={data.controls.season_type}
              onChange={(value) => {
                setSeasonType(value);
                setWeek(1);
                setGameId(null);
              }}
            />
            <SelectBox
              label="Week"
              options={data.controls.weeks}
              value={data.controls.week}
              onChange={(value) => {
                setWeek(Number(value));
                setGameId(null);
              }}
            />
          </Columns>

          {data.stopped ? null : (
            <>
              <TextInput
                label="Search teams"
                value={search}
                placeholder="e.g. Alabama, Ohio State…"
                onChange={(value) => setSearch(value)}
              />
              <SelectBox
                label="Select game"
                options={data.controls.games}
                value={data.controls.selected_label}
                onChange={(label) => setGameId(data.controls.game_options[label])}
              />
              <Caption>{data.controls.game_id_caption}</Caption>

              {data.figure ? (
                <>
                  <MetricRow metrics={data.metrics} />
                  <Divider />
                  <PlotlyChart figure={data.figure} height={480} />
                  <Expander label={data.raw_expander.label}>
                    <DataFrame table={data.raw_expander} height={data.raw_expander.height} />
                  </Expander>
                </>
              ) : null}
            </>
          )}
        </>
      )}
    </PageState>
  );
}
