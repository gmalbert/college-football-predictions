const SNAPSHOTS = {
  "/meta": "meta",
  "/home": "home",
  "/value-bets": "value-bets",
  "/team-explorer": "team-explorer",
  "/historical-analysis": "historical-analysis",
  "/model-performance": "model-performance",
  "/win-probability": "win-probability",
  "/preseason-outlook": "preseason-outlook",
  "/data-quality": "data-quality",
  "/total-market-signals": "total-market-signals",
};

const WEEKLY_SORTS = [
  "Edge (High→Low)",
  "Win Prob",
  "Matchup (A–Z)",
  "Kickoff (earliest first)",
];

let manifestPromise;

const CONFERENCE_TZ = {
  "America/New_York": "ET",
  "America/Chicago": "CT",
  "America/Denver": "MT",
  "America/Los_Angeles": "PT",
  "America/Anchorage": "AKT",
  "Pacific/Honolulu": "HT",
  UTC: "UTC",
};

async function fetchJson(path, signal) {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Static data request failed (${response.status}): ${path}`);
  return response.json();
}

function getManifest(signal) {
  if (!manifestPromise) {
    manifestPromise = fetchJson("/data/snapshots/manifest.json", signal).catch((error) => {
      manifestPromise = null;
      throw error;
    });
  }
  return manifestPromise;
}

function kickoffLabel(iso, timezone) {
  if (!iso) return "—";
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  }).formatToParts(new Date(iso));
  const part = (type) => parts.find((entry) => entry.type === type)?.value || "";
  return `${part("month")} ${part("day")} · ${part("hour")}:${part("minute")} ${part("dayPeriod")}`;
}

function applyWeeklySelection(source, params, timezone, weeksForSeason) {
  const season = Number(params.season ?? source.options.season);
  const week = Number(params.week ?? source.options.default_week);
  const conference = params.conference || "All";
  const minEdge = Number(params.min_edge ?? 0);
  const sortBy = WEEKLY_SORTS.includes(params.sort_by) ? params.sort_by : WEEKLY_SORTS[0];
  const metadata = source.static_rows || {};
  const gameKey = (row) => row.Game;
  let rows = [...(source.table?.rows || [])].filter((row) => {
    const info = metadata[gameKey(row)];
    if (!info) return false;
    if (conference !== "All" && !info.conferences.includes(conference)) return false;
    return info.edge === null || info.edge >= minEdge;
  });

  const indexed = rows.map((row, index) => ({ row, index }));
  indexed.sort((left, right) => {
    const a = metadata[gameKey(left.row)] || {};
    const b = metadata[gameKey(right.row)] || {};
    let order = 0;
    if (sortBy === "Edge (High→Low)") order = (b.edge ?? -Infinity) - (a.edge ?? -Infinity);
    else if (sortBy === "Win Prob") order = (b.win_prob ?? -Infinity) - (a.win_prob ?? -Infinity);
    else if (sortBy === "Matchup (A–Z)") order = gameKey(left.row).localeCompare(gameKey(right.row));
    else order = (a.start_date || "9999").localeCompare(b.start_date || "9999");
    return order || left.index - right.index;
  });
  rows = indexed.map(({ row }) => row);

  const utcColumn = source.table?.columns?.find((column) => column === "Kickoff (UTC)");
  const zoneLabel = CONFERENCE_TZ[timezone] || timezone.split("/").at(-1).replaceAll("_", " ");
  const kickoffColumn = `Kickoff (${zoneLabel})`;
  if (utcColumn && kickoffColumn !== utcColumn) {
    rows = rows.map((row) => {
      const info = metadata[gameKey(row)];
      const { [utcColumn]: _oldKickoff, ...rest } = row;
      return { ...rest, [kickoffColumn]: kickoffLabel(info?.start_date, timezone) };
    });
  }
  const columns = (source.table?.columns || []).map((column) =>
    column === utcColumn ? kickoffColumn : column,
  );
  const conferences = source.selection?.conferences || ["All"];

  return {
    ...source,
    options: {
      ...source.options,
      weeks: weeksForSeason,
    },
    selection: { ...source.selection, season, week, conference, min_edge: minEdge, sort_by: sortBy, conferences },
    game_count_line: `**${rows.length} games** — Season ${season} · Week ${week}`,
    empty_info: rows.length ? null : "No games match the current filters.",
    table_caption: `Compact view — kickoff times shown in ${zoneLabel}. Use the table’s horizontal scroll to see every market and recommendation.`,
    table: { columns, rows },
    table_height: Math.min(640, Math.max(140, 36 + 35 * rows.length)),
  };
}

function historicalH2h(rows, teamA, teamB) {
  const matchups = rows
    .filter(
      (game) =>
        (game.home_team === teamA && game.away_team === teamB) ||
        (game.home_team === teamB && game.away_team === teamA),
    )
    .sort((a, b) => Number(b.season) - Number(a.season));
  if (!matchups.length) {
    return {
      heading: "Head-to-Head Lookup",
      info: `No matchups found between ${teamA} and ${teamB} in the dataset.`,
    };
  }

  const aWins = matchups.filter((game) => {
    const margin = Number(game.home_margin);
    return (
      (game.home_team === teamA && margin > 0) ||
      (game.away_team === teamA && margin < 0)
    );
  }).length;
  const tableRows = matchups.map((game) => {
    const aHome = game.home_team === teamA;
    const aScore = aHome ? game.home_score : game.away_score;
    const bScore = aHome ? game.away_score : game.home_score;
    let result = "—";
    if (aScore !== null && aScore !== undefined) {
      result = `${teamA} ${aScore}–${bScore} ${teamB}`;
    }
    const spread =
      game.market_spread === null || game.market_spread === undefined
        ? Number.NaN
        : Number(game.market_spread);
    return {
      Season: Number(game.season),
      Week: Number(game.week || 0),
      Site: game.neutral_site ? "Neutral" : aHome ? "Home" : "Away",
      Result: result,
      Spread: Number.isFinite(spread) ? `${spread >= 0 ? "+" : ""}${spread.toFixed(1)}` : "—",
    };
  });

  return {
    heading: "Head-to-Head Lookup",
    metrics: [
      { label: `${teamA} wins`, value: String(aWins), delta: null, help: null },
      { label: `${teamB} wins`, value: String(matchups.length - aWins), delta: null, help: null },
      { label: "Games", value: String(matchups.length), delta: null, help: null },
    ],
    table: {
      columns: ["Season", "Week", "Site", "Result", "Spread"],
      rows: tableRows,
    },
  };
}

function applyValueBetSelection(source, params, timezone, catalog) {
  const season = Number(params.season ?? catalog.default_season);
  const weeks = catalog.weeks_by_season[String(season)] || [];
  const defaultWeek = catalog.default_week_by_season[String(season)] ?? catalog.default_week;
  const requestedWeek = Number(params.week ?? defaultWeek);
  const week = weeks.includes(requestedWeek) ? requestedWeek : defaultWeek;
  const betType = params.bet_type || "Spread";
  const minEdge = Number(params.min_edge ?? 2);
  const minConfidence = params.min_conf || "MODERATE";
  const sortBy = WEEKLY_SORTS.includes(params.sort_by) ? params.sort_by : WEEKLY_SORTS[0];
  const rowsByKey = source.static_rows || [];
  const metadata = new Map(rowsByKey.map((item) => [item.key, item]));
  const getKey = (row) => `${row["Bet Type"]}|${row.Game}|${row.Week}`;
  const confidenceRank = { LEAN: 1, MODERATE: 2, STRONG: 3 };
  const threshold = minConfidence === "All" ? 0 : confidenceRank[minConfidence] || 0;
  const tierRank = (row) => confidenceRank[row["Edge Tier"]] || 0;
  let rows = [...(source.table?.rows || [])].filter((row) => {
    const info = metadata.get(getKey(row));
    return (
      info &&
      (betType === "All" || row["Bet Type"] === betType) &&
      info.edge >= minEdge &&
      tierRank(row) >= threshold
    );
  });
  const indexed = rows.map((row, index) => ({ row, index }));
  indexed.sort((left, right) => {
    const a = metadata.get(getKey(left.row));
    const b = metadata.get(getKey(right.row));
    let order = 0;
    if (sortBy === "Edge (High→Low)") order = b.edge - a.edge;
    else if (sortBy === "Win Prob") order = (b.win_prob ?? -Infinity) - (a.win_prob ?? -Infinity);
    else if (sortBy === "Matchup (A–Z)") order = left.row.Game.localeCompare(right.row.Game);
    else order = (a.start_date || "9999").localeCompare(b.start_date || "9999");
    return order || left.index - right.index;
  });
  rows = indexed.map(({ row }) => row);

  const utcColumn = source.table?.columns?.find((column) => column === "Kickoff (UTC)");
  const zoneLabel = CONFERENCE_TZ[timezone] || timezone.split("/").at(-1).replaceAll("_", " ");
  const kickoffColumn = `Kickoff (${zoneLabel})`;
  if (utcColumn && kickoffColumn !== utcColumn) {
    rows = rows.map((row) => {
      const info = metadata.get(getKey(row));
      const { [utcColumn]: _oldKickoff, ...rest } = row;
      return { ...rest, [kickoffColumn]: kickoffLabel(info?.start_date, timezone) };
    });
  }

  const wins = rows.filter((row) => row.Result === "✅ WIN").length;
  const losses = rows.filter((row) => row.Result === "❌ LOSS").length;
  const settled = wins + losses;
  const selectedControls = {
    ...source.controls,
    seasons: catalog.seasons,
    season,
    weeks,
    week,
    default_week: defaultWeek,
    bet_type: betType,
    min_edge: minEdge,
    min_conf: minConfidence,
    sort_by: sortBy,
    start_bankroll: Number(params.start_bankroll ?? 1000),
    stake_method: params.stake_method || "Flat (1%)",
    bet_odds: Number(params.bet_odds ?? -110),
    scenario_probability: Number(params.scenario_probability ?? 0.5),
  };
  const metrics = [
    { label: "Total Bets", value: String(rows.length), delta: null, help: null },
    { label: "Record", value: settled ? `${wins}–${losses}` : "N/A", delta: null, help: null },
    { label: "Win Rate", value: settled ? `${((wins / settled) * 100).toFixed(1)}%` : "—", delta: null, help: null },
    { label: "Strong Edge Tier", value: String(rows.filter((row) => row["Edge Tier"] === "STRONG").length), delta: null, help: null },
  ];

  if (!rows.length) {
    return {
      ...source,
      controls: selectedControls,
      stopped: true,
      info: `No bets found with edge ≥ ${minEdge} pts (moneyline: pp) and confidence ≥ ${minConfidence}.`,
    };
  }

  const startBankroll = selectedControls.start_bankroll;
  const odds = selectedControls.bet_odds;
  const probability = selectedControls.scenario_probability;
  const stakeMethod = selectedControls.stake_method;
  const decimalOdds = odds < 0 ? 100 / Math.abs(odds) : odds / 100;
  const kelly = odds === 0 ? 0 : Math.max(0, Math.min((decimalOdds * probability - (1 - probability)) / decimalOdds, 0.25));
  const stakeFraction = stakeMethod === "Half Kelly" ? kelly / 2 : stakeMethod === "Full Kelly" ? kelly : 0.01;
  const curve = [startBankroll];
  let balance = startBankroll;
  rows.forEach((row) => {
    if (row.Result !== "✅ WIN" && row.Result !== "❌ LOSS") return;
    const stake = stakeFraction * balance;
    if (row.Result === "✅ WIN") balance += stake * decimalOdds;
    else balance -= stake;
    curve.push(balance);
  });
  const finalBalance = curve[curve.length - 1];
  const net = finalBalance - startBankroll;
  const money = (value) => Math.round(value).toLocaleString("en-US");
  const roi = startBankroll ? net / startBankroll : 0;
  const chart = curve.length > 1
    ? {
        figure: {
          data: [{ type: "scatter", y: curve, mode: "lines", line: { color: "#D4001C", width: 2 }, fill: "tozeroy", fillcolor: "rgba(212,0,28,0.1)", name: "Bankroll" }],
          layout: {
            title: { text: "Cumulative Bankroll" },
            xaxis: { title: { text: "Bet #" } },
            yaxis: { title: { text: "Bankroll ($)" } },
            shapes: [{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: startBankroll, y1: startBankroll, line: { dash: "dash", color: "gray" } }],
            annotations: [{ x: 1, xref: "paper", y: startBankroll, yref: "y", text: "Starting bankroll", showarrow: false }],
            paper_bgcolor: "#F7FBFF",
            plot_bgcolor: "#F7FBFF",
            font: { color: "#1A2B3C" },
          },
        },
        metrics: [
          { label: "Final Bankroll", value: `$${money(finalBalance)}`, delta: `${roi >= 0 ? "+" : ""}${(roi * 100).toFixed(1)}%`, help: null },
          { label: "Net P&L", value: `${net >= 0 ? "+" : "−"}$${money(Math.abs(net))}`, delta: null, help: null },
        ],
      }
    : null;

  return {
    ...source,
    controls: selectedControls,
    stopped: false,
    metrics,
    table: { columns: source.table.columns.map((column) => column === utcColumn ? kickoffColumn : column), rows },
    chart,
  };
}

export async function staticApiGet(path, params = {}, signal) {
  if (path === "/weekly-predictions") {
    const manifest = await getManifest(signal);
    const catalog = manifest.weekly_catalog;
    if (!catalog) throw new Error("The static weekly catalog has not been generated yet.");
    const season = Number(params.season ?? catalog.default_season);
    const weeks = catalog.weeks_by_season[String(season)] || [];
    const requestedWeek = Number(params.week ?? catalog.default_week);
    const week = weeks.includes(requestedWeek) ? requestedWeek : (weeks.includes(catalog.default_week) ? catalog.default_week : weeks[0]);
    if (week === undefined) throw new Error(`No static weekly data is available for season ${season}.`);
    const source = await fetchJson(`/data/snapshots/weekly/${season}-${week}.json`, signal);
    return applyWeeklySelection(
      source,
      { ...params, season, week },
      params.tz || "UTC",
      weeks,
    );
  }

  if (path === "/team-explorer") {
    const catalog = (await getManifest(signal)).team_catalog;
    if (!catalog?.teams?.length) return fetchJson("/data/snapshots/team-explorer.json", signal);
    const team = params.team || catalog.default_team;
    const season = Number(params.season ?? catalog.default_season);
    const url = catalog.snapshots?.[team]?.[String(season)];
    if (!url) throw new Error(`No static Team Explorer data exists for ${team}, ${season}.`);
    return fetchJson(url, signal);
  }

  if (path === "/win-probability") {
    const catalog = (await getManifest(signal)).win_probability_catalog;
    const seasons = catalog.seasons.map(Number);
    const season = seasons.includes(Number(params.season))
      ? Number(params.season)
      : catalog.default_season;
    const seasonType = catalog.season_types.includes(params.season_type)
      ? params.season_type
      : catalog.default_season_type;
    const weekOptions = seasonType === "Regular"
      ? Array.from({ length: 17 }, (_, index) => index + 1)
      : Array.from({ length: 5 }, (_, index) => index + 1);
    const requestedWeek = Number(params.week ?? catalog.default_week);
    const week = weekOptions.includes(requestedWeek) ? requestedWeek : weekOptions[0];
    const entry = catalog.schedule?.[String(season)]?.[seasonType]?.[String(week)];
    if (!entry) throw new Error(`No static Win Probability schedule exists for ${season} ${seasonType} week ${week}.`);

    const search = String(params.search || "").trim().toLocaleLowerCase();
    const games = entry.games.filter((label) => !search || label.toLocaleLowerCase().includes(search));
    const gameId = Number(params.game_id);
    const selectedLabel = games.find((label) => Number(entry.game_options[label]) === gameId) || games[0];
    const controls = {
      seasons: catalog.seasons,
      season,
      season_types: catalog.season_types,
      season_type: seasonType,
      weeks: weekOptions,
      week,
      search: String(params.search || ""),
      games,
      game_options: entry.game_options,
      selected_label: selectedLabel || "",
      game_id: selectedLabel ? entry.game_options[selectedLabel] : null,
      game_id_caption: selectedLabel ? `Game ID: \`${entry.game_options[selectedLabel]}\`` : "",
    };

    if (!selectedLabel) {
      const fallbackUrl = entry.empty_url || entry.snapshots[String(entry.game_options[entry.games[0]])];
      const empty = await fetchJson(fallbackUrl, signal);
      const noSearchMatches = entry.games.length > 0 && Boolean(search);
      return {
        ...empty,
        controls,
        warnings: noSearchMatches ? [`No games matching '${params.search}'.`] : empty.warnings,
        stopped: true,
      };
    }

    const selectedId = String(entry.game_options[selectedLabel]);
    const url = entry.snapshots[selectedId];
    if (!url) throw new Error(`No static Win Probability data exists for game ${selectedId}.`);
    const payload = await fetchJson(url, signal);
    return { ...payload, controls };
  }

  if (path === "/value-bets") {
    const catalog = (await getManifest(signal)).value_bets_catalog;
    if (!catalog?.seasons?.length) return fetchJson("/data/snapshots/value-bets.json", signal);
    const season = Number(params.season ?? catalog.default_season);
    const weeks = catalog.weeks_by_season[String(season)] || [];
    const defaultWeek = catalog.default_week_by_season[String(season)] ?? catalog.default_week;
    const requestedWeek = Number(params.week ?? defaultWeek);
    const week = weeks.includes(requestedWeek) ? requestedWeek : defaultWeek;
    const url = catalog.snapshots?.[String(season)]?.[String(week)];
    if (!url) throw new Error(`No static Value Bets data exists for ${season}, week ${week}.`);
    const source = await fetchJson(url, signal);
    return applyValueBetSelection(source, { ...params, season, week }, params.tz || "UTC", catalog);
  }

  if (path === "/preseason-outlook") {
    const catalog = (await getManifest(signal)).preseason_catalog;
    if (!catalog?.seasons?.length) return fetchJson("/data/snapshots/preseason-outlook.json", signal);
    const season = Number(params.season ?? catalog.default_season);
    const conference = params.conference || "All";
    const url = catalog.snapshots?.[String(season)]?.[conference];
    if (!url) throw new Error(`No static Preseason Outlook data exists for ${season}, ${conference}.`);
    return fetchJson(url, signal);
  }

  if (path === "/total-market-signals") {
    const catalog = (await getManifest(signal)).total_signals_catalog;
    if (!catalog?.seasons?.length) return fetchJson("/data/snapshots/total-market-signals.json", signal);
    const season = Number(params.season ?? catalog.default_season);
    const url = catalog.snapshots?.[String(season)];
    if (!url) throw new Error(`No static Total Market Signals data exists for ${season}.`);
    return fetchJson(url, signal);
  }

  if (path === "/historical-analysis") {
    const catalog = (await getManifest(signal)).historical_catalog;
    if (!catalog?.seasons?.length) return fetchJson("/data/snapshots/historical-analysis.json", signal);
    const seasonFrom = Number(params.season_from ?? catalog.default_season_from);
    const seasonTo = Number(params.season_to ?? catalog.default_season_to);
    const teamA = params.team_a || catalog.default_team_a;
    const teamB = params.team_b || catalog.default_team_b;
    const url = catalog.snapshots?.[String(seasonFrom)]?.[String(seasonTo)];
    if (!url) throw new Error(`No static historical data exists for ${seasonFrom}–${seasonTo}.`);

    const [payload, gameData] = await Promise.all([
      fetchJson(url, signal),
      fetchJson(catalog.games_url, signal),
    ]);
    const tabs = payload.tabs.map((tab) =>
      tab.id === "h2h" ? { ...tab, ...historicalH2h(gameData.rows || [], teamA, teamB) } : tab,
    );
    return {
      ...payload,
      selection: { ...payload.selection, season_from: seasonFrom, season_to: seasonTo, team_a: teamA, team_b: teamB },
      tabs,
    };
  }

  const snapshot = SNAPSHOTS[path];
  if (!snapshot) throw new Error(`No static snapshot is configured for ${path}.`);
  const payload = await fetchJson(`/data/snapshots/${snapshot}.json`, signal);
  if (Object.values(params).some((value) => value !== undefined && value !== null && value !== "")) {
    throw new Error(`Static data for ${path} does not yet include the requested selection.`);
  }
  return payload;
}

