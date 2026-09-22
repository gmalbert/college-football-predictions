/** Shared data-fetching hook with abort-on-unmount and re-fetch on param change. */
import { useEffect, useRef, useState } from "react";
import { apiGet } from "./api";

export function useApi(path, params) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const key = JSON.stringify(params || {});

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setState((previous) => ({ ...previous, loading: true, error: null }));

    apiGet(path, JSON.parse(key), controller.signal)
      .then((data) => {
        if (active) setState({ data, error: null, loading: false });
      })
      .catch((error) => {
        if (!active || error.name === "AbortError") return;
        setState({ data: null, error: error.message, loading: false });
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [path, key]);

  return state;
}

/** Read the ``theme`` query-string override used by the parity harness. */
export function useThemeOverride() {
  const ref = useRef(null);
  if (ref.current === null) {
    const value = new URLSearchParams(window.location.search).get("theme");
    ref.current = value === "night" || value === "day" ? value : "auto";
  }
  return ref.current;
}
