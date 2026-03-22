const TEAM_NAMES = {
  nba: {
    ATL: "Atlanta Hawks", BOS: "Boston Celtics", BKN: "Brooklyn Nets", CHA: "Charlotte Hornets",
    CHI: "Chicago Bulls", CLE: "Cleveland Cavaliers", DAL: "Dallas Mavericks", DEN: "Denver Nuggets",
    DET: "Detroit Pistons", GSW: "Golden State Warriors", HOU: "Houston Rockets", IND: "Indiana Pacers",
    LAC: "LA Clippers", LAL: "Los Angeles Lakers", MEM: "Memphis Grizzlies", MIA: "Miami Heat",
    MIL: "Milwaukee Bucks", MIN: "Minnesota Timberwolves", NOP: "New Orleans Pelicans", NYK: "New York Knicks",
    OKC: "Oklahoma City Thunder", ORL: "Orlando Magic", PHI: "Philadelphia 76ers", PHX: "Phoenix Suns",
    POR: "Portland Trail Blazers", SAC: "Sacramento Kings", SAS: "San Antonio Spurs", TOR: "Toronto Raptors",
    UTA: "Utah Jazz", WAS: "Washington Wizards",
  },
  mlb: {
    ARI: "Arizona Diamondbacks", ATL: "Atlanta Braves", BAL: "Baltimore Orioles", BOS: "Boston Red Sox",
    CHC: "Chicago Cubs", CWS: "Chicago White Sox", CIN: "Cincinnati Reds", CLE: "Cleveland Guardians",
    COL: "Colorado Rockies", DET: "Detroit Tigers", HOU: "Houston Astros", KCR: "Kansas City Royals",
    KC: "Kansas City Royals", LAA: "Los Angeles Angels", LAD: "Los Angeles Dodgers", MIA: "Miami Marlins",
    MIL: "Milwaukee Brewers", MIN: "Minnesota Twins", NYM: "New York Mets", NYY: "New York Yankees",
    OAK: "Oakland Athletics", ATH: "Athletics", PHI: "Philadelphia Phillies", PIT: "Pittsburgh Pirates",
    SDP: "San Diego Padres", SD: "San Diego Padres", SFG: "San Francisco Giants", SF: "San Francisco Giants",
    SEA: "Seattle Mariners", STL: "St. Louis Cardinals", TBR: "Tampa Bay Rays", TB: "Tampa Bay Rays",
    TEX: "Texas Rangers", TOR: "Toronto Blue Jays", WSN: "Washington Nationals", WAS: "Washington Nationals",
  },
  ncaa_baseball: {
    ABILCH: "Abilene Christian", AIRFOR: "Air Force", AKRON: "Akron", ALA: "Alabama",
    ALAM: "Alabama A&M", ALAST: "Alabama St.", ALBANY: "UAlbany", ALCORN: "Alcorn",
    APEAY: "Austin Peay", APPST: "App State", ARIZ: "Arizona", ARK: "Arkansas",
    ARKPB: "Ark.-Pine Bluff", ARKST: "Arkansas St.", ARMY: "Army West Point", AUBURN: "Auburn",
    AZST: "Arizona St.", BALDWA: "Baldwin Wallace", BALLST: "Ball St.", BAYLOR: "Baylor",
    BC: "Boston College", BECOOK: "Bethune-Cookman", BELLAR: "Bellarmine", BELMNT: "Belmont",
    BGSU: "Bowling Green", BINGHA: "Binghamton", BRAD: "Bradley", BRIARC: "Briar Cliff (IA)",
    BRITISHC: "British Colum. (CAN)", BROWN: "Brown", BRYANT: "Bryant", BUCKNL: "Bucknell",
    BUTLER: "Butler", BYU: "BYU", CAL: "California", CALBAP: "California Baptist",
    CALPLY: "Cal Poly", CAMPBL: "Campbell", CANISI: "Canisius", CARK: "Central Ark.",
    CCONN: "Central Conn. St.", CHAMIN: "Chaminade", CHAR: "Charlotte", CHARSO: "Charleston So.",
    CHBAP: "Champion Chris.", CINCY: "Cincinnati", CITDEL: "The Citadel", CKATL: "Clark Atlanta",
    CLEM: "Clemson", CMICH: "Central Mich.", COCAR: "Coastal Carolina", COFC: "Col. of Charleston",
    COLUMB: "Columbia", COMESA: "Colorado Mesa", COPPIN: "Coppin St.", CORN: "Cornell",
    CREIGH: "Creighton", CSFULL: "Cal St. Fullerton", CSUBAK: "CSU Bakersfield", CSUN: "CSUN",
    CSUSB: "CSUSB", DART: "Dartmouth", DAVID: "Davidson", DAYTON: "Dayton",
    DBU: "DBU", DEL: "Delaware", DELST: "Delaware St.", DILLRD: "Dillard",
    DUKE: "Duke", ECAR: "East Carolina", EILL: "Eastern Ill.", EKY: "Eastern Ky.",
    ELON: "Elon", EMICH: "Eastern Mich.", ETSU: "ETSU", EVANS: "Evansville",
    FAIR: "Fairfield", FAU: "Fla. Atlantic", FDU: "FDU", FGCU: "FGCU",
    FINDLY: "Findlay", FIU: "FIU", FLA: "Florida", FLAM: "Florida A&M",
    FORDHM: "Fordham", FREPAC: "Fresno Pacific", FRESNO: "Fresno St.", FSU: "Florida St.",
    GASOU: "Ga. Southern", GAST: "Georgia St.", GATECH: "Georgia Tech", GCANYN: "Grand Canyon",
    GMU: "George Mason", GONZ: "Gonzaga", GRAMB: "Grambling", GTOWN: "Georgetown",
    GWASH: "George Washington", GWEBB: "Gardner-Webb", HARV: "Harvard", HAWAII: "Hawaii",
    HAWPAC: "Hawaii Pacific", HIGHPT: "High Point", HOFSTR: "Hofstra", HOLYCR: "Holy Cross",
    HOU: "Houston", HOUCHR: "Houston Christian", HOUSTONV: "Houston-Victoria", HUSTIL: "Huston-Tillotson",
    ILL: "Illinois", ILLST: "Illinois St.", INCWRD: "UIW", IND: "Indiana",
    INDST: "Indiana St.", IONA: "Iona", IOWA: "Iowa", JACKST: "Jackson St.",
    JAXST: "Jacksonville St.", JMU: "James Madison", JVILLE: "Jacksonville", KANSAS: "Kansas",
    KANST: "Kansas St.", KENSAW: "Kennesaw St.", KENT: "Kent St.", KYST: "Kentucky St.",
    LA: "Louisiana", LACHRIST: "La. Christian", LAFAYE: "Lafayette", LAMAR: "Lamar University",
    LAMON: "ULM", LASALL: "La Salle", LATECH: "Louisiana Tech", LBSU: "Long Beach St.",
    LEHIGH: "Lehigh", LEMOYN: "Le Moyne", LIBRTY: "Liberty", LINWOD: "Lindenwood",
    LIPSCO: "Lipscomb", LIU: "LIU", LMU: "LMU (CA)", LONGWD: "Longwood",
    LORAS: "Loras", LOUIS: "Louisville", LSU: "LSU", LSUALX: "LSU-Alexandria",
    MAINE: "Maine", MANHAT: "Manhattan", MARIST: "Marist", MARSH: "Marshall",
    MCNEES: "McNeese", MD: "Maryland", MEM: "Memphis", MERCER: "Mercer",
    MERCYH: "Mercyhurst", MERMCK: "Merrimack", MIAMI: "Miami (FL)", MIAOH: "Miami (OH)",
    MICH: "Michigan", MICHST: "Michigan St.", MIDTEN: "Middle Tenn.", MILWKE: "Milwaukee",
    MINN: "Minnesota", MISS: "Ole Miss", MISSST: "Mississippi St.", MIZZOU: "Missouri",
    MONMTH: "Monmouth", MOREST: "Morehead St.", MOST: "Missouri St.", MSVAL: "Mississippi Val.",
    MTSTMY: "Mount St. Mary's", MUHLEN: "Muhlenberg", MURRAY: "Murray St.", NALA: "North Ala.",
    NAVY: "Navy", NCAT: "N.C. A&T", NCOLO: "Northern Colo.", NCST: "NC State",
    NDAME: "Notre Dame", NDST: "North Dakota St.", NEB: "Nebraska", NEVADA: "Nevada",
    NEWHAV: "New Haven", NIAGRA: "Niagara", NICHST: "Nicholls", NIU: "NIU",
    NJIT: "NJIT", NKY: "Northern Ky.", NMEX: "New Mexico", NMST: "New Mexico St.",
    NOEAST: "Northeastern", NOFLA: "North Florida", NORFLK: "Norfolk St.", NOVA: "Villanova",
    NW: "Northwestern", NWST: "Northwestern St.", OAK: "Oakland", OAKLANDC: "Oakland City",
    ODU: "Old Dominion", OHIO: "Ohio", OHIOST: "Ohio St.", OKLA: "Oklahoma",
    OKLAST: "Oklahoma St.", OMAHA: "Omaha", OREGN: "Oregon", OREST: "Oregon St.",
    ORU: "Oral Roberts", PACIF: "Pacific", PENN: "Penn", PENNST: "Penn St.",
    PEPPER: "Pepperdine", PITT: "Pittsburgh", PORT: "Portland", PRESBY: "Presbyterian",
    PRINCE: "Princeton", PURDUE: "Purdue", PVAM: "Prairie View", QUENNC: "Queens (NC)",
    QUINN: "Quinnipiac", RADFRD: "Radford", RICE: "Rice", RICH: "Richmond",
    RIDER: "Rider", RUST: "Rust", RUTGER: "Rutgers", SACHRT: "Sacred Heart",
    SACST: "Sacramento St.", SALA: "South Alabama", SAMFRD: "Samford", SAMHOU: "Sam Houston",
    SCAR: "South Carolina", SCUPS: "USC Upstate", SDAKST: "South Dakota St.", SDSU: "San Diego St.",
    SEATTL: "Seattle U", SELA: "Southeastern La.", SEMO: "Southeast Mo. St.", SETON: "Seton Hall",
    SFA: "SFA", SFRAN: "San Francisco", SIENA: "Siena", SIND: "Southern Ind.",
    SIU: "Southern Ill.", SIUE: "SIUE", SJSU: "San Jose St.", SMISS: "Southern Miss.",
    SOUNO: "Southern-N.O.", SOUTHEAS: "Southeastern Bapt.", SOUTHR: "Southern U.", STAMBROS: "St. Ambrose",
    STAN: "Stanford", STBONA: "St. Bonaventure", STCLAR: "Santa Clara", STETSN: "Stetson",
    STILLMAN: "Stillman", STJOES: "Saint Joseph's", STJOHN: "St. John's (NY)", STLOU: "Saint Louis",
    STMART: "Saint Martin's", STMARY: "Saint Mary's (CA)", STONEH: "Stonehill", STONY: "Stony Brook",
    STPTR: "Saint Peter's", STTHOM: "St. Thomas (MN)", TAMUCC: "A&M-Corpus Christi", TARLET: "Tarleton St.",
    TBA: "TBA", TCU: "TCU", TENN: "Tennessee", TEXAS: "Texas",
    TIFFIN: "Tiffin", TNTECH: "Tennessee Tech", TOLEDO: "Toledo", TOUGLO: "Tougaloo",
    TOWSON: "Towson", TROY: "Troy", TULANE: "Tulane", TXAM: "Texas A&M",
    TXARL: "UT Arlington", TXSOU: "Texas Southern", TXST: "Texas St.", TXTECH: "Texas Tech",
    UAB: "UAB", UALR: "Little Rock", UCDAV: "UC Davis", UCF: "UCF",
    UCIRV: "UC Irvine", UCLA: "UCLA", UCONN: "UConn", UCRIV: "UC Riverside",
    UCSB: "UC Santa Barbara", UCSD: "UC San Diego", UGA: "Georgia", UIC: "UIC",
    UK: "Kentucky", UMASS: "Massachusetts", UMASSL: "UMass Lowell", UMBC: "UMBC",
    UMES: "UMES", UNC: "North Carolina", UNCA: "UNC Asheville", UNCG: "UNC Greensboro",
    UNCW: "UNCW", UNIOTN: "Union (TN)", UNLV: "UNLV", UNO: "New Orleans",
    UOFSW: "Southwest (NM)", URI: "Rhode Island", USC: "Southern California", USD: "San Diego",
    USF: "South Fla.", UTAH: "Utah", UTMAR: "UT Martin", UTRGV: "UTRGV",
    UTSA: "UTSA", UTTECH: "Utah Tech", UTVAL: "Utah Valley", UVA: "Virginia",
    VALPO: "Valparaiso", VANDY: "Vanderbilt", VCU: "VCU", VMI: "VMI",
    VT: "Virginia Tech", WAGNER: "Wagner", WAKE: "Wake Forest", WASH: "Washington",
    WASHST: "Washington St.", WCAR: "Western Caro.", WCHES: "West Chester", WESTCLIF: "Westcliff",
    WGA: "West Ga.", WICHST: "Wichita St.", WILEY: "Wiley", WILL: "Western Ill.",
    WINTHR: "Winthrop", WKY: "Western Ky.", WMICH: "Western Mich.", WMMRY: "William & Mary",
    WOFFRD: "Wofford", WRIGHT: "Wright St.", WRTBRG: "Wartburg", WSHBRN: "Washburn",
    WVU: "West Virginia", XAVIER: "Xavier", XAVLA: "Xavier (LA)", YALE: "Yale",
    YSU: "Youngstown St.",
  },
  nhl: {
    ANA: "Anaheim Ducks", BOS: "Boston Bruins", BUF: "Buffalo Sabres", CAR: "Carolina Hurricanes",
    CBJ: "Columbus Blue Jackets", CGY: "Calgary Flames", CHI: "Chicago Blackhawks", COL: "Colorado Avalanche",
    DAL: "Dallas Stars", DET: "Detroit Red Wings", EDM: "Edmonton Oilers", FLA: "Florida Panthers",
    LAK: "Los Angeles Kings", MIN: "Minnesota Wild", MTL: "Montreal Canadiens", NJD: "New Jersey Devils",
    NSH: "Nashville Predators", NYI: "New York Islanders", NYR: "New York Rangers", OTT: "Ottawa Senators",
    PHI: "Philadelphia Flyers", PIT: "Pittsburgh Penguins", SEA: "Seattle Kraken", SJS: "San Jose Sharks",
    STL: "St. Louis Blues", TBL: "Tampa Bay Lightning", TOR: "Toronto Maple Leafs", UTA: "Utah Hockey Club",
    VAN: "Vancouver Canucks", VGK: "Vegas Golden Knights", WPG: "Winnipeg Jets", WSH: "Washington Capitals",
  },
  kbo: {
    KIA: "KIA Tigers", DOO: "Doosan Bears", HAN: "Hanwha Eagles", LOT: "Lotte Giants", LG: "LG Twins",
    NCD: "NC Dinos", KTW: "KT Wiz", SAM: "Samsung Lions", SSG: "SSG Landers", KIW: "Kiwoom Heroes",
  },
  liga_mx: {
    AME: "Club America", TOL: "Toluca", TIG: "Tigres UANL", MTY: "Monterrey", PUM: "Pumas UNAM",
    CHI: "Guadalajara", CRU: "Cruz Azul", PAC: "Pachuca", NEC: "Necaxa", SAN: "Atletico San Luis",
    JUA: "Juarez", LEO: "Leon", MAZ: "Mazatlan", PUE: "Puebla", QRO: "Queretaro",
    ATS: "Atlas", TJU: "Tijuana", SNT: "Santos Laguna",
  },
  laliga: {
    RMA: "Real Madrid", FCB: "Barcelona", ATM: "Atletico Madrid", ATH: "Athletic Club",
    BET: "Real Betis", VIL: "Villarreal", RSO: "Real Sociedad", SEV: "Sevilla",
    VAL: "Valencia", GIR: "Girona", OSA: "Osasuna", CEL: "Celta Vigo",
    GET: "Getafe", ALA: "Alaves", MLL: "Mallorca", RAY: "Rayo Vallecano",
    ESP: "Espanyol", VLL: "Real Valladolid", LEG: "Leganes", LPA: "Las Palmas",
  },
  euroleague: {
    ALB: "ALBA Berlin", ASM: "AS Monaco", BAS: "Baskonia", BAR: "FC Barcelona", BBN: "Bayern Munich",
    BER: "ALBA Berlin", BOL: "Virtus Bologna", DUB: "Dubai BC", CZV: "Crvena Zvezda",
    EFE: "Anadolu Efes", FBB: "Fenerbahce", HTA: "Maccabi Tel Aviv", MIL: "EA7 Emporio Armani Milan",
    OLY: "Olympiacos", PAN: "Panathinaikos", PAR: "Paris Basketball", PBB: "Partizan Belgrade",
    RMB: "Real Madrid", ZAL: "Zalgiris Kaunas", VAL: "Valencia Basket", VIR: "Virtus Bologna",
  },
};

export function getTeamDisplayName(sportKey, teamCodeOrName) {
  const raw = String(teamCodeOrName || "").trim();
  if (!raw) return raw;

  const sport = String(sportKey || "").toLowerCase();
  const map = TEAM_NAMES[sport] || {};
  const key = raw.toUpperCase();

  return map[key] || raw;
}

export function expandTeamCodeInText(sportKey, text) {
  const raw = String(text || "").trim();
  if (!raw) return raw;

  const sport = String(sportKey || "").toLowerCase();
  const map = TEAM_NAMES[sport] || {};
  const upper = raw.toUpperCase();

  if (map[upper]) {
    return map[upper];
  }

  const firstSpace = raw.indexOf(" ");
  if (firstSpace > 0) {
    const first = raw.slice(0, firstSpace).toUpperCase();
    if (map[first]) {
      return `${map[first]}${raw.slice(firstSpace)}`;
    }
  }

  return raw;
}
