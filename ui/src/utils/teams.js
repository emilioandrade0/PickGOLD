function isGenericSideLabel(value) {
  const v = String(value || "").trim().toLowerCase();
  return v === "home" || v === "away" || v === "home win" || v === "away win";
}

function parseTeamsFromGameName(gameName) {
  const text = String(gameName || "");
  const match = text.match(/^\s*([^@]+?)\s*@\s*([^@]+?)\s*$/);
  if (!match) {
    return { awayTeam: "", homeTeam: "" };
  }
  return {
    awayTeam: match[1].trim(),
    homeTeam: match[2].trim(),
  };
}

export function resolveEventTeams(event) {
  const fallback = parseTeamsFromGameName(event?.game_name);

  const awayRaw = String(event?.away_team || "").trim();
  const homeRaw = String(event?.home_team || "").trim();

  const awayTeam = awayRaw && !isGenericSideLabel(awayRaw) ? awayRaw : fallback.awayTeam;
  const homeTeam = homeRaw && !isGenericSideLabel(homeRaw) ? homeRaw : fallback.homeTeam;

  return {
    awayTeam: awayTeam || "AWAY",
    homeTeam: homeTeam || "HOME",
  };
}

export function resolveSidePick(pick, teams) {
  const text = String(pick || "").trim();
  if (!text) {
    return text;
  }

  const normalized = text.toLowerCase();
  if (normalized === "home" || normalized === "home win" || normalized === "local" || normalized === "1") {
    return teams.homeTeam;
  }
  if (normalized === "away" || normalized === "away win" || normalized === "visitante" || normalized === "2") {
    return teams.awayTeam;
  }

  return text;
}
