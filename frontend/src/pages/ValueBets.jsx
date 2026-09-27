/** Value Bets — mirrors ``pages/2_Value_Bets.py``. */
import { useEffect, useMemo, useState } from "react";
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import TableDownloads from "../components/TableDownloads";
import { PlotlyChart } from "../components/LazyPlotlyChart";
import { browserTimezone } from "../api";
import {
  Alert,
  Caption,
  Columns,
  DataFrame,
  Divider,
  Heading,
  MetricRow,
  NumberInput,
  SelectBox,
  Slider,
} from "../components/ui";
import { useApi } from "../useApi";

const SORT_OPTIONS = [
  "Edge (High→Low)",
  "Win Prob",
  "Matchup (A–Z)",
  "Kickoff (earliest first)",
];

export default function ValueBets() {
  const [season, setSeason] = useState(null);
  const [week, setWeek] = useState(null);
  const [betType, setBetType] = useState("Spread");
  const [minEdge, setMinEdge] = useState(2.0);
  const [minConf, setMinConf] = useState("MODERATE");
  const [sortBy, setSortBy] = useState(SORT_OPTIONS[0]);
  const [startBankroll, setStartBankroll] = useState(1000);
  const [stakeMethod, setStakeMethod] = useState("Flat (1%)");
  const [betOdds, setBetOdds] = useState(-110);
  const [scenarioProbability, setScenarioProbability] = useState(0.5);

  const params = useMemo(
    () => ({
      season,
      week,
      bet_type: betType,
      min_edge: minEdge,
      min_conf: minConf,
      sort_by: sortBy,
      tz: browserTimezone(),
      start_bankroll: startBankroll,
      stake_method: stakeMethod,
      bet_odds: betOdds,
      scenario_probability: scenarioProbability,
    }),
    [season, week, betType, minEdge, minConf, sortBy, startBankroll, stakeMethod, betOdds, scenarioProbability]
  );
  const state = useApi("/value-bets", params);
  const controls = state.data?.controls;

  useEffect(() => {
    if (!controls) return;
    if (season === null) setSeason(controls.season);
    if (week === null || !controls.weeks.includes(week)) setWeek(controls.week);
  }, [controls, season, week]);

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

          {data.controls ? (
            <>
              <Columns>
                <SelectBox
                  label="Season"
                  options={data.controls.seasons}
                  value={data.controls.season}
                  onChange={(value) => {
                    setSeason(Number(value));
                    setWeek(null);
                  }}
                />
                <SelectBox
                  label="Week"
                  options={data.controls.weeks.map((value) => ({
                    value,
                    label: `Week ${value}`,
                  }))}
                  value={data.controls.week}
                  onChange={(value) => setWeek(Number(value))}
                />
                <SelectBox
                  label="Bet Type"
                  options={data.controls.bet_types}
                  value={data.controls.bet_type}
                  onChange={(value) => setBetType(value)}
                />
                <Slider
                  label="Min Edge (pts)"
                  min={0.5}
                  max={10}
                  step={0.5}
                  value={data.controls.min_edge}
                  onChange={(value) => setMinEdge(value)}
                />
                <SelectBox
                  label="Min Edge Tier"
                  options={data.controls.min_conf_options}
                  value={data.controls.min_conf}
                  onChange={(value) => setMinConf(value)}
                />
                <SelectBox
                  label="Sort By"
                  options={SORT_OPTIONS}
                  value={data.controls.sort_by}
                  onChange={(value) => setSortBy(value)}
                />
              </Columns>

              {(data.captions || []).map((text) => (
                <Caption key={text}>{text}</Caption>
              ))}

              {data.stopped ? (
                data.info ? <Alert kind="info">{data.info}</Alert> : null
              ) : (
                <>
                  <MetricRow metrics={data.metrics} />
                  <Divider />
                  <div className="table-section-heading">
                    <Heading level={3}>{data.table_heading}</Heading>
                    <TableDownloads
                      table={data.table}
                      title="Value Bets"
                      subtitle={`${data.controls.season} · Week ${data.controls.week} · ${data.controls.bet_type} · ${data.controls.min_edge} pt minimum edge · ${data.controls.min_conf} · ${data.controls.sort_by}`}
                      filename={`value-bets-${data.controls.season}-week-${data.controls.week}-${data.controls.bet_type.toLowerCase()}`}
                    />
                  </div>
                  <DataFrame table={data.table} />

                  <Divider />
                  <Heading level={3}>{data.simulator_heading}</Heading>
                  <Columns>
                    <NumberInput
                      label="Starting Bankroll ($)"
                      value={data.controls.start_bankroll}
                      min={100}
                      step={100}
                      onChange={(value) => setStartBankroll(value)}
                    />
                    <SelectBox
                      label="Stake Method"
                      options={data.controls.stake_methods}
                      value={data.controls.stake_method}
                      onChange={(value) => setStakeMethod(value)}
                    />
                    <NumberInput
                      label="Odds (American)"
                      value={data.controls.bet_odds}
                      step={5}
                      onChange={(value) => setBetOdds(value)}
                    />
                    <Slider
                      label="Scenario Win Probability"
                      min={0.5}
                      max={0.65}
                      step={0.01}
                      value={data.controls.scenario_probability}
                      onChange={(value) => setScenarioProbability(value)}
                    />
                  </Columns>
                  <Caption>{data.simulator_caption}</Caption>

                  {data.chart ? (
                    <>
                      <PlotlyChart figure={data.chart.figure} />
                      <MetricRow metrics={data.chart.metrics} />
                    </>
                  ) : (
                    <Caption>{data.chart_empty_caption}</Caption>
                  )}
                </>
              )}
            </>
          ) : data.info ? (
            <Alert kind="info">{data.info}</Alert>
          ) : null}
          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
