const FALLBACK_LOGO = '/logos/default-team.svg';

const MLB_LOGO_OVERRIDES: Record<string, string> = {
  AZ: 'ari',
  ARI: 'ari',
  WSH: 'wsh',
  WSN: 'wsh',
  CWS: 'chw',
  CHW: 'chw',
  KC: 'kc',
  KCR: 'kc',
  SD: 'sd',
  SDP: 'sd',
  SF: 'sf',
  SFG: 'sf',
  TB: 'tb',
  TBR: 'tb',
};

const NBA_ALIASES: Record<string, string> = {
  GS: 'GSW',
  NO: 'NOP',
  SA: 'SAS',
  UT: 'UTA',
  WS: 'WAS',
  PHO: 'PHX',
  BK: 'BKN',
  NJ: 'BKN',
  NY: 'NYK',
};

const NBA_LOGO_OVERRIDES: Record<string, string> = {
  ATL: 'atl',
  BOS: 'bos',
  BKN: 'bkn',
  CHA: 'cha',
  CHI: 'chi',
  CLE: 'cle',
  DAL: 'dal',
  DEN: 'den',
  DET: 'det',
  GSW: 'gs',
  HOU: 'hou',
  IND: 'ind',
  LAC: 'lac',
  LAL: 'lal',
  MEM: 'mem',
  MIA: 'mia',
  MIL: 'mil',
  MIN: 'min',
  NOP: 'no',
  NYK: 'ny',
  OKC: 'okc',
  ORL: 'orl',
  PHI: 'phi',
  PHX: 'phx',
  POR: 'por',
  SAC: 'sac',
  SAS: 'sa',
  TOR: 'tor',
  UTA: 'utah',
  WAS: 'wsh',
};

const LIGA_MX_ESPN_CODES: Record<string, string> = {
  AME: '227',
  ATS: '216',
  CAZ: '218',
  CHI: '219',
  GDL: '219',
  JUA: '17851',
  LEO: '228',
  MAZ: '20702',
  MTY: '220',
  NEC: '229',
  NCX: '229',
  PAC: '234',
  PUE: '231',
  PUM: '233',
  QRO: '222',
  SAN: '225',
  SLA: '15720',
  SLP: '15720',
  ASL: '15720',
  TIG: '232',
  UANL: '232',
  TIJ: '10125',
  TOL: '223',
  UNAM: '233',
};

const LALIGA_ESPN_CODES: Record<string, string> = {
  ALA: '96',
  ALV: '96',
  ATH: '93',
  ATM: '1068',
  BAR: '83',
  BET: '244',
  CEL: '85',
  ESP: '88',
  GET: '2922',
  GIR: '9813',
  LEG: '17500',
  LEV: '216',
  MLL: '84',
  MLC: '84',
  OSA: '97',
  RAY: '87',
  RMA: '86',
  RSO: '89',
  SEV: '243',
  VAL: '94',
  VIL: '102',
  VLL: '95',
};

const BUNDESLIGA_ESPN_CODES: Record<string, string> = {
  FCB: '132', BAY: '132',
  BVB: '124', DOR: '124',
  B04: '131', LEV: '131',
  RBL: '11420',
  SGE: '125', FRA: '125',
  SCF: '126', FRE: '126',
  M05: '127', MAI: '127',
  WOB: '129',
  VFB: '134', STU: '134',
  TSG: '7911', HOF: '7911',
  FCA: '384', AUG: '384',
  UBE: '598', FCU: '598',
  SVW: '137', BRE: '137',
  BMG: '268', MGL: '268',
  BOC: '121',
  HDH: '9672', HEI: '9672',
  STP: '270',
  KIE: '116',
};

const KBO_LOGO_OVERRIDES: Record<string, string> = {
  DOO: 'doosan-bears',
  HAN: 'hanwha-eagles',
  KIA: 'kia-tigers',
  KIW: 'kiwoom-heroes',
  KTW: 'kt-wiz',
  LG: 'lg-twins',
  LOT: 'lotte-giants',
  NCD: 'nc-dinos',
  SAM: 'samsung-lions',
  SSG: 'ssg-landers',
};

const TRIPLE_A_LOGO_IDS: Record<string, string> = {
  ABQ: '342',
  'ALBUQUERQUE ISOTOPES': '342',
  BUF: '422',
  'BUFFALO BISONS': '422',
  CLT: '494',
  'CHARLOTTE KNIGHTS': '494',
  COL: '445',
  'COLUMBUS CLIPPERS': '445',
  DUR: '234',
  'DURHAM BULLS': '234',
  ELP: '4904',
  'EL PASO CHIHUAHUAS': '4904',
  GWN: '431',
  'GWINNETT STRIPERS': '431',
  IND: '484',
  'INDIANAPOLIS INDIANS': '484',
  IOW: '451',
  'IOWA CUBS': '451',
  JAX: '564',
  'JACKSONVILLE JUMBO SHRIMP': '564',
  LV: '400',
  'LAS VEGAS AVIATORS': '400',
  LHV: '1410',
  'LEHIGH VALLEY IRONPIGS': '1410',
  LOU: '416',
  'LOUISVILLE BATS': '416',
  MEM: '235',
  'MEMPHIS REDBIRDS': '235',
  NAS: '556',
  'NASHVILLE SOUNDS': '556',
  NOR: '568',
  'NORFOLK TIDES': '568',
  OKC: '238',
  'OKLAHOMA CITY COMETS': '238',
  OMA: '541',
  'OMAHA STORM CHASERS': '541',
  RNO: '2310',
  'RENO ACES': '2310',
  ROC: '534',
  'ROCHESTER RED WINGS': '534',
  RR: '102',
  'ROUND ROCK EXPRESS': '102',
  SAC: '105',
  'SACRAMENTO RIVER CATS': '105',
  SL: '561',
  'SALT LAKE BEES': '561',
  SWB: '531',
  'SCRANTON/WILKES-BARRE RAILRIDERS': '531',
  STP: '1960',
  'ST. PAUL SAINTS': '1960',
  SUG: '5434',
  'SUGAR LAND SPACE COWBOYS': '5434',
  SYR: '552',
  'SYRACUSE METS': '552',
  TAC: '529',
  'TACOMA RAINIERS': '529',
  TOL: '512',
  'TOLEDO MUD HENS': '512',
  WOR: '533',
  'WORCESTER RED SOX': '533',
};

function normalizeAbbr(abbr: string | null | undefined): string {
  return String(abbr || '').trim().toUpperCase();
}

function normalizeNBAAbbr(abbr: string | null | undefined): string {
  const normalized = normalizeAbbr(abbr);
  return NBA_ALIASES[normalized] || normalized;
}

export function getMLBLogoUrl(abbr: string | null | undefined): string {
  const normalized = normalizeAbbr(abbr);
  const code = MLB_LOGO_OVERRIDES[normalized] || normalized.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/mlb/500/${code}.png`;
}

export function getNBALogoUrl(abbr: string | null | undefined): string {
  const normalized = normalizeNBAAbbr(abbr);
  const code = NBA_LOGO_OVERRIDES[normalized];
  return code ? `https://a.espncdn.com/i/teamlogos/nba/500/${code}.png` : FALLBACK_LOGO;
}

export function getNHLLogo(teamCode: string | null | undefined): string {
  const code = normalizeAbbr(teamCode).toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nhl/500/${code}.png`;
}

export function getLigaMXLogo(teamCode: string | null | undefined): string {
  const normalized = normalizeAbbr(teamCode);
  const code = LIGA_MX_ESPN_CODES[normalized] || normalized.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/soccer/500/${code}.png`;
}

export function getLigaMXLogoUrl(abbr: string | null | undefined): string {
  return getLigaMXLogo(abbr);
}

export function getLaLigaLogo(teamCode: string | null | undefined): string {
  const normalized = normalizeAbbr(teamCode);
  const code = LALIGA_ESPN_CODES[normalized] || normalized.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/soccer/500/${code}.png`;
}

export function getBundesligaLogo(teamCode: string | null | undefined): string {
  const normalized = normalizeAbbr(teamCode);
  const code = BUNDESLIGA_ESPN_CODES[normalized] || normalized.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/soccer/500/${code}.png`;
}

export function getKBOLogo(teamCode: string | null | undefined): string {
  const normalized = normalizeAbbr(teamCode);
  const code = KBO_LOGO_OVERRIDES[normalized] || normalized.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/baseball/500/${code}.png`;
}

export function getTripleALogo(teamCode: string | null | undefined): string {
  const normalized = normalizeAbbr(teamCode);
  const id = TRIPLE_A_LOGO_IDS[normalized];
  return id ? `https://www.mlbstatic.com/team-logos/${id}.svg` : FALLBACK_LOGO;
}

export function getTeamLogoUrl(sportKey: string, abbr: string | null | undefined): string | null {
  if (sportKey === 'mlb') return getMLBLogoUrl(abbr);
  if (sportKey === 'nba') return getNBALogoUrl(abbr);
  if (sportKey === 'nhl') return getNHLLogo(abbr);
  if (sportKey === 'liga_mx') return getLigaMXLogo(abbr);
  if (sportKey === 'laliga') return getLaLigaLogo(abbr);
  if (sportKey === 'bundesliga') return getBundesligaLogo(abbr);
  if (sportKey === 'kbo') return getKBOLogo(abbr);
  if (sportKey === 'triple_a') return getTripleALogo(abbr);
  return null;
}

export { FALLBACK_LOGO };
