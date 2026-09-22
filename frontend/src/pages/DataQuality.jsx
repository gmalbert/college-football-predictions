/** Data & Model Quality — mirrors ``pages/9_Data_Quality.py``. */
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import { Alert, Caption, DataFrame, Heading, MetricRow } from "../components/ui";
import { useApi } from "../useApi";

export default function DataQuality() {
  const state = useApi("/data-quality", {});
  return (
    <PageState state={state}>
      {(data) => (
        <>
          <Heading level={1}>{data.title}</Heading>
          <Caption>{data.caption}</Caption>

          <MetricRow metrics={data.metrics} />

          {(data.captions || []).map((text) => (
            <Caption key={text}>{text}</Caption>
          ))}
          {(data.warnings || []).map((text) => (
            <Alert key={text} kind="warning">
              {text}
            </Alert>
          ))}

          {data.table ? <DataFrame table={data.table} /> : null}
          <Alert kind="info">{data.info}</Alert>

          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
