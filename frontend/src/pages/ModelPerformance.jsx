/** Model Performance — mirrors ``pages/5_Model_Performance.py``. */
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import { PlotlyChart } from "../components/LazyPlotlyChart";
import {
  Alert,
  Caption,
  DataFrame,
  Divider,
  Expander,
  Heading,
  MetricRow,
} from "../components/ui";
import { useApi } from "../useApi";

function CatalogTable({ rows }) {
  if (!rows || !rows.length) return null;
  return (
    <table className="markdown-table">
      <thead>
        <tr>
          <th>Feature</th>
          <th>Description</th>
          <th>Why helpful</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row[0]}>
            {row.map((cell, index) => (
              <td key={index}>{cell}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ModelPerformance() {
  const state = useApi("/model-performance", {});
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
          {(data.errors || []).map((text) => (
            <Alert key={text} kind="error">
              {text}
            </Alert>
          ))}

          {data.stopped ? null : (
            <>
              <MetricRow metrics={data.metrics} />
              <Caption>{data.comparison_caption}</Caption>
              <DataFrame table={data.comparison} />

              <Expander label={data.gates_expander.label}>
                {data.gates_expander.info ? (
                  <Alert kind="info">{data.gates_expander.info}</Alert>
                ) : (
                  <DataFrame table={data.gates_expander.table} />
                )}
              </Expander>

              <Divider />

              {data.calibration.figure ? (
                <>
                  <Heading level={3}>{data.calibration.heading}</Heading>
                  <PlotlyChart figure={data.calibration.figure} />
                </>
              ) : data.calibration.info ? (
                <Alert kind="info">{data.calibration.info}</Alert>
              ) : null}

              {data.calibration.figure ? <Divider /> : null}

              {data.ats_week.figure ? (
                <>
                  <Heading level={3}>{data.ats_week.heading}</Heading>
                  <PlotlyChart figure={data.ats_week.figure} />
                  <Divider />
                </>
              ) : null}

              <Heading level={3}>{data.spread_importance.heading}</Heading>
              {data.spread_importance.figure ? (
                <>
                  <PlotlyChart figure={data.spread_importance.figure} />
                  <Expander label={data.spread_importance.catalog.label}>
                    <Alert kind="info">{data.spread_importance.catalog.info}</Alert>
                    <CatalogTable rows={data.spread_importance.catalog.rows} />
                  </Expander>
                </>
              ) : (
                <Alert kind="info">{data.spread_importance.info}</Alert>
              )}

              <Divider />

              <Heading level={3}>{data.total_importance.heading}</Heading>
              {data.total_importance.figure ? (
                <>
                  <PlotlyChart figure={data.total_importance.figure} />
                  <Expander label={data.total_importance.catalog.label}>
                    <Alert kind="info">{data.total_importance.catalog.info}</Alert>
                    <CatalogTable rows={data.total_importance.catalog.rows} />
                  </Expander>
                </>
              ) : (
                <Alert kind="info">{data.total_importance.info}</Alert>
              )}

              <Divider />
              <Heading level={3}>{data.summary_heading}</Heading>
              <MetricRow metrics={data.summary_metrics} />
            </>
          )}
          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
