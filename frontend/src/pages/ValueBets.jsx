/** Value Bets — mirrors ``pages/2_Value_Bets.py``. */
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
  NumberInput,
  SelectBox,
  Slider,
} from "../components/ui";
import { useApi } from "../useApi";

export default function ValueBets() {
  const [season, setSeason] = useState(null);
  const [betType, setBetType] = useState("Spread");
  const [minEdge, setMinEdge] = useState(2.0);
  const [minConf, setMinConf] = useState("MODERATE");
  const [startBankroll, setStartBankroll] = useState(1000);
  const [stakeMethod, setStakeMethod] = useState("Flat (1%)");
  const [betOdds, setBetOdds] = useState(-110);
  const [scenarioProbability, setScenarioProbability] = useState(0.5);

  const params = useMemo(
    () => ({
      season,
      bet_type: betType,
      min_edge: minEdge,
      min_conf: minConf,
      start_bankroll: startBankroll,
      stake_method: stakeMethod,
      bet_odds: betOdds,
      scenario_probability: scenarioProbability,
    }),
    [season, betType, minEdge, minConf, startBankroll, stakeMethod, betOdds, scenarioProbability]
  );
  const state = useApi("/value-bets", params);
  const controls = state.data?.controls;

  useEffect(() => {
    if (controls && season === null) setSeason(controls.season);
  }, [controls, season]);

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

          {data.stopped ? (
            data.info ? (
              <Alert kind="info">{data.info}</Alert>
            ) : null
          ) : (
            <>
              <Columns>
                <SelectBox
                  label="Season"
                  options={data.controls.seasons}
                  value={data.controls.season}
                  onChange={(value) => setSeason(Number(value))}
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
              </Columns>

              {(data.captions || []).map((text) => (
                <Caption key={text}>{text}</Caption>
              ))}

              <MetricRow metrics={data.metrics} />
              <Divider />
              <Heading level={3}>{data.table_heading}</Heading>
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
          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
