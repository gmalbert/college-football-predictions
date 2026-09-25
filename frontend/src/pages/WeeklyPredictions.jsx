/** Weekly Predictions — mirrors ``pages/1_Weekly_Predictions.py``. */
import { useEffect, useMemo, useState } from "react";
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import {
  Alert,
  Caption,
  Columns,
  DataFrame,
  Divider,
  Expander,
  Heading,
  Markdown,
  MetricRow,
  SelectBox,
  Slider,
} from "../components/ui";
import { weeklyParams } from "../api";
import { useApi } from "../useApi";

const SORT_OPTIONS = ["Edge (High→Low)", "Win Prob", "Game"];

export default function WeeklyPredictions() {
  const [season, setSeason] = useState(null);
  const [week, setWeek] = useState(null);
  const [conference, setConference] = useState("All");
  const [minEdge, setMinEdge] = useState(0.0);
  const [sortBy, setSortBy] = useState(SORT_OPTIONS[0]);

  const params = useMemo(
    () => weeklyParams({ season, week, conference, min_edge: minEdge, sort_by: sortBy }),
    [season, week, conference, minEdge, sortBy]
  );
  const state = useApi("/weekly-predictions", params);
  const options = state.data?.options;

  // Adopt the server's default season/week on first load.
  useEffect(() => {
    if (!options) return;
    if (season === null) setSeason(options.season);
    if (week === null) setWeek(options.default_week);
  }, [options, season, week]);

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
          {(data.infos || []).map((text) => (
            <Alert key={text} kind="info">
              {text}
            </Alert>
          ))}
          {data.stopped ? null : (
            <>
              <Columns>
                <SelectBox
                  label="Season"
                  options={data.options.seasons}
                  value={data.selection.season}
                  onChange={(value) => setSeason(Number(value))}
                />
                <SelectBox
                  label="Week"
                  options={data.options.weeks.map((value) => ({
                    value,
                    label: `Week ${value}`,
                  }))}
                  value={data.selection.week}
                  onChange={(value) => setWeek(Number(value))}
                />
              </Columns>
              <Caption>{data.selection.season_caption}</Caption>

              <Columns>
                <SelectBox
                  label="Conference"
                  options={data.selection.conferences}
                  value={data.selection.conference}
                  onChange={(value) => setConference(value)}
                />
                <Slider
                  label="Min Edge (spread or O/U pts)"
                  min={0}
                  max={10}
                  step={0.5}
                  value={data.selection.min_edge}
                  onChange={(value) => setMinEdge(value)}
                />
                <SelectBox
                  label="Sort By"
                  options={SORT_OPTIONS}
                  value={data.selection.sort_by}
                  onChange={(value) => setSortBy(value)}
                />
              </Columns>

              <Markdown>{data.game_count_line}</Markdown>
              {(data.captions || []).map((text) => (
                <Caption key={text}>{text}</Caption>
              ))}
              <Divider />

              {data.empty_info ? (
                <Alert kind="info">{data.empty_info}</Alert>
              ) : (
                <>
                  {data.parlay?.expander ? (
                    <Expander label={data.parlay.expander.label}>
                      <Caption>{data.parlay.expander.caption}</Caption>
                      <DataFrame table={data.parlay.expander.table} />
                    </Expander>
                  ) : data.parlay?.caption ? (
                    <Caption>{data.parlay.caption}</Caption>
                  ) : null}

                  <Caption>{data.table_caption}</Caption>
                  <DataFrame table={data.table} height={data.table_height} />
                </>
              )}
            </>
          )}
          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
