/**
 * Streamlit-compatible UI primitives.
 *
 * Each component mirrors the DOM Streamlit emits (``data-testid``-ish class
 * names: stMetric, stCaption, stAlert, stDataFrame, stTabs, stExpander …) so
 * that a single CSS file can reproduce the Frost theme for both apps and the
 * Playwright parity harness can query the same selectors on either side.
 */
import { useState } from "react";

/* ------------------------------------------------------------------ metrics */

/** Streamlit tints a delta green/red and prefixes ↑/↓ when it starts with a number. */
function deltaTone(delta) {
  if (typeof delta !== "string") return "";
  const match = /^\s*([+-]?\d+(?:\.\d+)?)/.exec(delta);
  if (!match) return "";
  const value = Number(match[1]);
  if (!Number.isFinite(value) || value === 0) return "";
  return value > 0 ? " positive" : " negative";
}

export function Metric({ label, value, delta, help }) {
  return (
    <div className="stMetric" data-testid="stMetric" title={help || undefined}>
      <p className="stMetricLabel" data-testid="stMetricLabel">
        {label}
      </p>
      <p className="stMetricValue" data-testid="stMetricValue">
        {value}
      </p>
      {delta ? (
        <p className={`stMetricDelta${deltaTone(delta)}`} data-testid="stMetricDelta">
          {delta}
        </p>
      ) : null}
    </div>
  );
}

export function MetricRow({ metrics }) {
  if (!metrics || !metrics.length) return null;
  return (
    <div className="metric-row" data-testid="stHorizontalBlock">
      {metrics.map((metric) => (
        <Metric key={metric.label} {...metric} />
      ))}
    </div>
  );
}

/**
 * Vertically stacked metrics.
 *
 * ``st.metric`` calls made one after another inside a single ``st.columns()``
 * column stack, they do not sit side by side — that only happens when each
 * metric is placed in its own column.
 */
export function MetricStack({ metrics }) {
  if (!metrics || !metrics.length) return null;
  return (
    <div className="metric-stack">
      {metrics.map((metric) => (
        <Metric key={metric.label} {...metric} />
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- captions */

export function Caption({ children }) {
  return (
    <div className="stCaption" data-testid="stCaptionContainer">
      {typeof children === "string" ? renderInline(children) : children}
    </div>
  );
}

/* ------------------------------------------------------------------ alerts */

const ALERT_ICONS = { warning: "⚠️", info: "ℹ️", error: "🚫", success: "✅" };

export function Alert({ kind = "info", children }) {
  return (
    <div
      className={`stAlert ${kind}`}
      data-testid="stAlert"
      data-alert-kind={kind}
      role="alert"
    >
      <span className="stAlert-icon">{ALERT_ICONS[kind] || ALERT_ICONS.info}</span>
      <div className="stAlert-body">
        {typeof children === "string" ? renderInline(children) : children}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- markdown */

function renderInline(text) {
  // Handles the small subset of Markdown the pages emit: **bold**, `code`,
  // *italic* and the two-space hard line break.
  const nodes = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) nodes.push(text.slice(lastIndex, match.index));
    const token = match[0];
    if (token.startsWith("**")) {
      nodes.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      nodes.push(<code key={key++}>{token.slice(1, -1)}</code>);
    } else {
      nodes.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }
    lastIndex = match.index + token.length;
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes;
}

export function Markdown({ children, className = "" }) {
  if (children === null || children === undefined) return null;
  const text = String(children);
  const lines = text.split(/\n/);
  return (
    <div className={`stMarkdown ${className}`} data-testid="stMarkdownContainer">
      {lines.map((line, index) => {
        if (line.trim() === "---") return <hr key={index} />;
        const hardBreak = line.endsWith("  ");
        const content = hardBreak ? line.slice(0, -2) : line;
        if (content.trim() === "") return <br key={index} />;
        return (
          <p key={index}>
            {renderInline(content)}
            {hardBreak ? <br /> : null}
          </p>
        );
      })}
    </div>
  );
}

export function Heading({ level = 1, children, className = "" }) {
  const Tag = `h${level}`;
  return (
    <Tag className={`stHeading ${className}`.trim()} data-testid="stHeading">
      {children}
    </Tag>
  );
}

/* --------------------------------------------------------------- dataframes */

function formatCell(value, columnConfig, columnName) {
  if (value === null || value === undefined) return "";
  const config = columnConfig?.[columnName];
  if (config?.format) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      if (config.format === "%.2f") return numeric.toFixed(2);
      if (config.format === "%.1f%%") return `${numeric.toFixed(1)}%`;
      if (config.format === "%.1f pts") return `${numeric.toFixed(1)} pts`;
    }
  }
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return String(Number(value.toFixed(6)));
  }
  return String(value);
}

export function DataFrame({ table, height, hideIndex = true }) {
  if (!table || !table.columns || !table.columns.length) {
    return <div className="df-empty">No data</div>;
  }
  const { columns, rows, column_config: columnConfig, progress_columns: progressColumns } = table;

  const style = height ? { maxHeight: `${height}px` } : undefined;

  return (
    <div className="stDataFrame" data-testid="stDataFrame">
      <div className="df-scroll" style={style}>
        <table className="df">
          {hideIndex ? null : null}
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column) => {
                  const raw = row[column];
                  const isProgress = Boolean(progressColumns?.[column]);
                  const numeric = Number(raw);
                  return (
                    <td key={column}>
                      {isProgress && Number.isFinite(numeric) ? (
                        <span className="progress-cell">
                          <span className="progress-track">
                            <span
                              className="progress-fill"
                              style={{ width: `${Math.max(0, Math.min(100, numeric))}%` }}
                            />
                          </span>
                          <span>{formatCell(raw, columnConfig, column)}</span>
                        </span>
                      ) : (
                        formatCell(raw, columnConfig, column)
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- tabs */

export function Tabs({ tabs, renderPanel }) {
  const [active, setActive] = useState(0);
  if (!tabs || !tabs.length) return null;
  return (
    <div className="stTabs" data-testid="stTabs">
      <div className="tablist" role="tablist">
        {tabs.map((tab, index) => (
          <button
            key={tab.id || index}
            role="tab"
            aria-selected={index === active}
            onClick={() => setActive(index)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>
      {/* Streamlit keeps every tab panel mounted and hides the inactive ones,
          so the React build does the same. */}
      {tabs.map((tab, index) => (
        <div
          key={tab.id || index}
          className="tabpanel"
          data-testid="stTabsContent"
          role="tabpanel"
          hidden={index !== active}
        >
          {renderPanel(tab, index)}
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- expander */

export function Expander({ label, expanded = false, children }) {
  return (
    <details className="stExpander" data-testid="stExpander" open={expanded}>
      <summary>{label}</summary>
      <div className="stExpander-body">{children}</div>
    </details>
  );
}

/* ----------------------------------------------------------------- widgets */

export function SelectBox({ label, options, value, onChange }) {
  return (
    <div className="stSelectbox" data-testid="stSelectbox">
      <label className="stWidgetLabel" data-testid="stWidgetLabel">{label}</label>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => {
          const optionValue = typeof option === "object" ? option.value : option;
          const optionLabel = typeof option === "object" ? option.label : option;
          return (
            <option key={String(optionValue)} value={optionValue}>
              {optionLabel}
            </option>
          );
        })}
      </select>
    </div>
  );
}

export function Slider({ label, min, max, step, value, onChange }) {
  return (
    <div className="stSlider" data-testid="stSlider">
      <label className="stWidgetLabel" data-testid="stWidgetLabel">
        {label}
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="stSlider-value">({Number(value).toFixed(1)})</span>
    </div>
  );
}

export function NumberInput({ label, value, onChange, min, step }) {
  return (
    <div className="stNumberInput" data-testid="stNumberInput">
      <label className="stWidgetLabel" data-testid="stWidgetLabel">{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

export function TextInput({ label, value, onChange, placeholder }) {
  return (
    <div className="stTextInput" data-testid="stTextInput">
      {label ? <label className="stWidgetLabel" data-testid="stWidgetLabel">{label}</label> : null}
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ layout */

export function Columns({ widths, children }) {
  const items = Array.isArray(children) ? children : [children];
  return (
    <div className="row" data-testid="stHorizontalBlock">
      {items.map((child, index) => (
        <div
          className="col"
          key={index}
          style={widths ? { flexGrow: widths[index], flexBasis: 0 } : undefined}
        >
          {child}
        </div>
      ))}
    </div>
  );
}

export function Divider() {
  return <hr data-testid="stDivider" />;
}
