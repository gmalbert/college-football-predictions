/** Home — mirrors ``predictions.py::home_page``. */
import Footer from "../components/Footer";
import PageState from "../components/PageState";
import {
  Alert,
  Caption,
  Columns,
  Divider,
  Heading,
  Markdown,
  MetricRow,
  MetricStack,
} from "../components/ui";
import { useApi } from "../useApi";

export default function Home() {
  const state = useApi("/home", {});
  return (
    <PageState state={state}>
      {(data) => (
        <>
          {data.logo ? (
            <Columns widths={[1, 4]}>
              <img src={data.logo} alt="logo" style={{ width: 120 }} />
              <div>
                <Heading level={1}>{data.title}</Heading>
                <Divider />
              </div>
            </Columns>
          ) : (
            <>
              <Heading level={1}>{data.title}</Heading>
              <Divider />
            </>
          )}

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
          {(data.captions || []).map((text) => (
            <Caption key={text}>{text}</Caption>
          ))}

          <Columns>
            {data.columns.map((column) => (
              <div key={column.heading}>
                <Heading level={3}>{column.heading}</Heading>
                {column.kind === "deltas" ? (
                  <>
                    {(column.items || []).map((item) => (
                      <Markdown key={item.title}>
                        {`**${item.title}**  \n${item.summary}`}
                      </Markdown>
                    ))}
                    {column.caption ? <Caption>{column.caption}</Caption> : null}
                  </>
                ) : column.layout === "stacked" ? (
                  <>
                    <MetricStack metrics={column.metrics} />
                    {column.caption ? <Caption>{column.caption}</Caption> : null}
                  </>
                ) : (
                  <>
                    <MetricRow metrics={column.metrics} />
                    {column.caption ? <Caption>{column.caption}</Caption> : null}
                  </>
                )}
              </div>
            ))}
          </Columns>

          {data.footer ? <Footer /> : null}
        </>
      )}
    </PageState>
  );
}
