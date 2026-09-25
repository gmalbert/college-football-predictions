/** Total Market Signals — mirrors ``pages/10_Total_Market_Signals.py``.
 *
 * The page exists in the repository but is not registered in
 * ``st.navigation``, so it is intentionally absent from the sidebar here too.
 */
import { useEffect, useState } from "react";
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import { Alert, Caption, DataFrame, Heading, MetricRow, SelectBox } from "../components/ui";
import { useApi } from "../useApi";

export default function TotalMarketSignals() {
  const [season, setSeason] = useState(null);
  const state = useApi("/total-market-signals", { season });
  const decisions = state.data?.decisions;

  useEffect(() => {
    if (decisions?.seasons?.length && season === null) setSeason(decisions.season);
  }, [decisions, season]);

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
          {(data.errors || []).map((text) => (
            <Alert key={text} kind="error">
              {text}
            </Alert>
          ))}

          {data.stopped ? null : (
            <>
              <MetricRow metrics={data.metrics} />
              <Caption>{data.strategy_caption}</Caption>

              <Heading level={3}>{data.decisions.heading}</Heading>
              {data.decisions.info ? (
                <Alert kind="info">{data.decisions.info}</Alert>
              ) : (
                <>
                  {data.decisions.seasons?.length ? (
                    <SelectBox
                      label="Held-out season"
                      options={data.decisions.seasons}
                      value={data.decisions.season}
                      onChange={(value) => setSeason(Number(value))}
                    />
                  ) : null}
                  <DataFrame
                    table={{
                      columns: data.decisions.columns,
                      rows: data.decisions.rows,
                    }}
                  />
                </>
              )}

              <Heading level={3}>{data.shadow.heading}</Heading>
              {data.shadow.info ? (
                <Alert kind="info">{data.shadow.info}</Alert>
              ) : (
                <DataFrame table={data.shadow.table} />
              )}
            </>
          )}

          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
