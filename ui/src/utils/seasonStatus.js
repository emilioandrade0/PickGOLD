const SPORT_SEASON_CONFIG = {
  nba: {
    cadence: "cross_year",
    seasonStart: { month: 10, day: 21, hour: 18, minute: 0 },
    seasonEnd: { month: 4, day: 12, hour: 23, minute: 59 },
    championBySeason: {
      "2023-24": "Boston Celtics",
      "2024-25": "Oklahoma City Thunder",
    },
    championFallback: "Pendiente de actualizar",
  },
  wnba: {
    cadence: "single_year",
    seasonStart: { month: 5, day: 8, hour: 18, minute: 0 },
    seasonEnd: { month: 9, day: 24, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "New York Liberty",
      "2025": "Las Vegas Aces",
    },
    championFallback: "Pendiente de actualizar",
  },
  mlb: {
    cadence: "single_year",
    seasonStart: { month: 3, day: 25, hour: 12, minute: 0 },
    seasonEnd: { month: 9, day: 27, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "Los Angeles Dodgers",
      "2025": "Los Angeles Dodgers",
    },
    championFallback: "Pendiente de actualizar",
  },
  lmb: {
    cadence: "single_year",
    seasonStart: { month: 4, day: 18, hour: 12, minute: 0 },
    seasonEnd: { month: 9, day: 14, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "Diablos Rojos del Mexico",
      "2025": "Diablos Rojos del Mexico",
    },
    championFallback: "Pendiente de actualizar",
  },
  triple_a: {
    cadence: "single_year",
    seasonStart: { month: 3, day: 28, hour: 12, minute: 0 },
    seasonEnd: { month: 9, day: 26, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "Omaha Storm Chasers",
      "2025": "Jacksonville Jumbo Shrimp",
    },
    championFallback: "Pendiente de actualizar",
  },
  ncaa_baseball: {
    cadence: "single_year",
    seasonStart: { month: 2, day: 13, hour: 12, minute: 0 },
    seasonEnd: { month: 6, day: 22, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "Tennessee",
      "2025": "LSU",
    },
    championFallback: "Pendiente de actualizar",
  },
  tennis: {
    cadence: "single_year",
    seasonStart: { month: 1, day: 1, hour: 9, minute: 0 },
    seasonEnd: { month: 11, day: 16, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "Jannik Sinner (ATP Finals)",
      "2025": "Jannik Sinner (ATP Finals)",
    },
    championFallback: "Pendiente de actualizar",
  },
  kbo: {
    cadence: "single_year",
    seasonStart: { month: 3, day: 28, hour: 12, minute: 0 },
    seasonEnd: { month: 9, day: 6, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "KIA Tigers",
      "2025": "LG Twins",
    },
    championFallback: "Pendiente de actualizar",
  },
  nhl: {
    cadence: "cross_year",
    seasonStart: { month: 10, day: 7, hour: 18, minute: 0 },
    seasonEnd: { month: 4, day: 16, hour: 23, minute: 59 },
    championBySeason: {
      "2023-24": "Florida Panthers",
      "2024-25": "Florida Panthers",
    },
    championFallback: "Pendiente de actualizar",
  },
  euroleague: {
    cadence: "cross_year",
    seasonStart: { month: 9, day: 30, hour: 12, minute: 0 },
    seasonEnd: { month: 4, day: 17, hour: 23, minute: 59 },
    championBySeason: {
      "2023-24": "Panathinaikos",
      "2024-25": "Fenerbahce Beko Istanbul",
    },
    championFallback: "Pendiente de actualizar",
  },
  liga_mx: {
    cadence: "single_year",
    seasonStart: { month: 1, day: 9, hour: 12, minute: 0 },
    seasonEnd: { month: 5, day: 24, hour: 23, minute: 59 },
    championBySeason: {
      "2024": "Club America",
      "2025": "Toluca",
    },
    championFallback: "Pendiente de actualizar",
  },
  laliga: {
    cadence: "cross_year",
    seasonStart: { month: 8, day: 15, hour: 12, minute: 0 },
    seasonEnd: { month: 5, day: 24, hour: 23, minute: 59 },
    championBySeason: {
      "2023-24": "Real Madrid",
      "2024-25": "FC Barcelona",
    },
    championFallback: "Pendiente de actualizar",
  },
  bundesliga: {
    cadence: "cross_year",
    seasonStart: { month: 8, day: 22, hour: 12, minute: 0 },
    seasonEnd: { month: 5, day: 16, hour: 23, minute: 59 },
    championBySeason: {
      "2023-24": "Bayer Leverkusen",
      "2024-25": "Bayern Munich",
    },
    championFallback: "Pendiente de actualizar",
  },
  ligue1: {
    cadence: "cross_year",
    seasonStart: { month: 8, day: 15, hour: 12, minute: 0 },
    seasonEnd: { month: 5, day: 16, hour: 23, minute: 59 },
    championBySeason: {
      "2023-24": "Paris Saint-Germain",
      "2024-25": "Paris Saint-Germain",
    },
    championFallback: "Pendiente de actualizar",
  },
};

const DATE_FORMATTER = new Intl.DateTimeFormat("es-MX", {
  weekday: "short",
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function buildDateAtYear(year, point) {
  const monthIndex = Math.max(1, Number(point?.month) || 1) - 1;
  const day = Math.max(1, Number(point?.day) || 1);
  const hour = Math.max(0, Number(point?.hour) || 0);
  const minute = Math.max(0, Number(point?.minute) || 0);
  return new Date(year, monthIndex, day, hour, minute, 0, 0);
}

function formatSeasonLabel(cadence, startDate, endDate) {
  if (cadence === "cross_year") {
    const endShort = String(endDate.getFullYear()).slice(-2);
    return `${startDate.getFullYear()}-${endShort}`;
  }
  return String(startDate.getFullYear());
}

function resolveSeasonWindow(config, now) {
  const currentYear = now.getFullYear();

  if (config.cadence === "cross_year") {
    const startThisYear = buildDateAtYear(currentYear, config.seasonStart);

    const seasonStart = now >= startThisYear
      ? startThisYear
      : buildDateAtYear(currentYear - 1, config.seasonStart);
    const seasonEnd = buildDateAtYear(seasonStart.getFullYear() + 1, config.seasonEnd);

    const inSeason = now >= seasonStart && now <= seasonEnd;
    const nextSeasonStart = inSeason
      ? buildDateAtYear(seasonStart.getFullYear() + 1, config.seasonStart)
      : (now < seasonStart
        ? seasonStart
        : buildDateAtYear(seasonStart.getFullYear() + 1, config.seasonStart));

    return { seasonStart, seasonEnd, nextSeasonStart, inSeason };
  }

  const startThisYear = buildDateAtYear(currentYear, config.seasonStart);
  const endThisYear = buildDateAtYear(currentYear, config.seasonEnd);
  const inSeason = now >= startThisYear && now <= endThisYear;

  const nextSeasonStart = inSeason
    ? buildDateAtYear(currentYear + 1, config.seasonStart)
    : (now < startThisYear ? startThisYear : buildDateAtYear(currentYear + 1, config.seasonStart));

  const seasonStart = inSeason
    ? startThisYear
    : (now < startThisYear ? buildDateAtYear(currentYear - 1, config.seasonStart) : startThisYear);
  const seasonEnd = inSeason
    ? endThisYear
    : (now < startThisYear ? buildDateAtYear(currentYear - 1, config.seasonEnd) : endThisYear);

  return { seasonStart, seasonEnd, nextSeasonStart, inSeason };
}

function resolveChampion(config, completedSeasonLabel) {
  const championBySeason = config.championBySeason || {};
  if (championBySeason[completedSeasonLabel]) {
    return {
      championName: championBySeason[completedSeasonLabel],
      championSeasonLabel: completedSeasonLabel,
    };
  }

  return {
    championName: config.championFallback || "Pendiente de actualizar",
    championSeasonLabel: completedSeasonLabel,
  };
}

function formatDateTime(value) {
  if (!(value instanceof Date) || Number.isNaN(value.getTime())) return "N/A";
  return DATE_FORMATTER.format(value);
}

export function resolveSeasonStatus(sportKey, sportLabel, now = new Date()) {
  const config = SPORT_SEASON_CONFIG[sportKey];
  if (!config) {
    return {
      sportLabel,
      inSeason: false,
      statusLabel: "Temporada sin calendario configurado",
      countdownLabel: "Proxima actualizacion",
      countdownTarget: null,
      countdownTargetText: "N/A",
      seasonLabel: "N/A",
      completedSeasonLabel: "N/A",
      championName: "Pendiente de actualizar",
      championSeasonLabel: "N/A",
      seasonStartText: "N/A",
      seasonEndText: "N/A",
      nextSeasonStartText: "N/A",
    };
  }

  const { seasonStart, seasonEnd, nextSeasonStart, inSeason } = resolveSeasonWindow(config, now);
  const seasonLabel = formatSeasonLabel(config.cadence, seasonStart, seasonEnd);

  let completedStart;
  let completedEnd;
  if (inSeason) {
    if (config.cadence === "cross_year") {
      completedStart = buildDateAtYear(seasonStart.getFullYear() - 1, config.seasonStart);
      completedEnd = buildDateAtYear(seasonEnd.getFullYear() - 1, config.seasonEnd);
    } else {
      completedStart = buildDateAtYear(seasonStart.getFullYear() - 1, config.seasonStart);
      completedEnd = buildDateAtYear(seasonEnd.getFullYear() - 1, config.seasonEnd);
    }
  } else {
    completedStart = seasonStart;
    completedEnd = seasonEnd;
  }

  const completedSeasonLabel = formatSeasonLabel(config.cadence, completedStart, completedEnd);
  const championInfo = resolveChampion(config, completedSeasonLabel);

  return {
    sportLabel,
    inSeason,
    statusLabel: inSeason ? "Temporada activa" : "Offseason",
    countdownLabel: inSeason ? "Termina en" : "Vuelve en",
    countdownTarget: inSeason ? seasonEnd : nextSeasonStart,
    countdownTargetText: formatDateTime(inSeason ? seasonEnd : nextSeasonStart),
    seasonLabel,
    completedSeasonLabel,
    championName: championInfo.championName,
    championSeasonLabel: championInfo.championSeasonLabel,
    seasonStartText: formatDateTime(seasonStart),
    seasonEndText: formatDateTime(seasonEnd),
    nextSeasonStartText: formatDateTime(nextSeasonStart),
  };
}

export function getCountdownParts(targetDate, now = new Date()) {
  if (!(targetDate instanceof Date) || Number.isNaN(targetDate.getTime())) {
    return { days: "00", hours: "00", minutes: "00", seconds: "00", expired: true };
  }

  const diffMs = Math.max(0, targetDate.getTime() - now.getTime());
  const totalSeconds = Math.floor(diffMs / 1000);
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return {
    days: String(days).padStart(2, "0"),
    hours: String(hours).padStart(2, "0"),
    minutes: String(minutes).padStart(2, "0"),
    seconds: String(seconds).padStart(2, "0"),
    expired: diffMs <= 0,
  };
}
