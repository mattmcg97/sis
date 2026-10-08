"""Search terms for the Reddit e-sports betting monitor."""

# eFootball player names. Most are ordinary words, so a name only counts
# when it appears in ALL CAPS or next to a betting/esports term.
PLAYERS = """KRAKEN SABRE RELIC JUDGEMENT ROCKET PHOENIX METEOR SAGE DREAD MERLIN
RICOCHET CLOCKWORK FUSE NIGHTMARE ENFORCER KILLJOY KATANA HERMES ACE ROMAN
ORIGINAL REGRET MAVERICK STEEL SPARK MINDFLAYER GADGET IMPERIAL SENTRY HURRICANE
PINNACLE KEVLAR MERCENARY PRISMATIC SCHOLAR PURSUIT MIRAGE SHARK CLASH STRIDER
LEGEND WARRIOR OFFICER HEADHUNTER MISSILE MEMENTO PRECISION SHROUD LUNA SHURIKEN
LIZARD VALIANT MERCURY FENRIR MIMIC SUMMIT LYNX CRYSTAL PSYCHIC MONSTER MIDAS
UNICORN LEO DESTINY GENIUS ASTRONAUT LAZARUS OVERDRIVE PATHFINDER OCTANE DON
NIRVANA AVATAR SENTINEL TOMBSTONE SORCERER ANVIL DAME SEEKER BLAZE NEON SPLINTER
AERO INVENTOR MARVEL EMBER PARADOX FRACTURE IDOL""".split()

# Sport terms: a post must mention at least one of these to be kept.
SPORT_TERMS = [
    "efootball", "e-football", "esoccer", "e-soccer", "ebasketball", "e-basketball",
    "enfl", "e-nfl", "madden sim", "esports football", "virtual football",
    "cyber football", "cyber basketball", "fifa 8 min", "2k sim", "nba 2k esports",
]

# Betting terms, weighted: totals/unders are priority 1.
BET_TERMS = {
    "under": 3, "unders": 3, "total": 3, "totals": 3, "o/u": 3, "over/under": 3,
    "total points": 3, "points total": 3,
    "moneyline": 2, "ml": 2, "spread": 2, "ats": 2, "handicap": 2,
    "tip": 1, "tips": 1, "pick": 1, "picks": 1, "lock": 1, "parlay": 1, "units": 1,
    "telegram": 2, "discord": 1, "vip": 2, "fixed": 3, "sure bet": 2,
}

BOOKS = ["fanduel", "fan duel", "hard rock", "hardrock", "hard rock bet",
         "draftkings", "bet365", "betano", "pinnacle", "betmgm"]

REGIONS = ["ontario", "brazil", "brasil", "belgium", "belgie", "belgique",
           "usa", "florida", "new jersey"]

# Reddit search queries (each returns at most ~1000 results).
QUERIES = [
    'efootball (bet OR betting OR tips OR picks OR under OR total)',
    'esoccer (bet OR betting OR tips OR picks OR under OR total)',
    'ebasketball (bet OR betting OR tips OR picks OR under OR total)',
    '"e-football" betting', '"e-soccer" betting', '"e-basketball" betting',
    '"cyber football" bet', '"cyber basketball" bet',
    'esports (fanduel OR "hard rock") (under OR total OR moneyline OR spread)',
    'efootball fanduel', 'efootball "hard rock"', 'esoccer telegram tips',
    'ebasketball telegram tips', 'esoccer apostas', 'ebasketball apostas',
]

# Subreddit-name searches for dedicated communities.
SUBREDDIT_QUERIES = [
    "esoccer", "ebasketball", "efootball", "esports betting", "esoccer betting",
    "ebasketball betting", "cyber football", "fifa betting", "betting tips",
    "apostas esports", "sports picks",
]

# Always crawl these (add any known tipster subs here).
WATCH_SUBREDDITS = ["sportsbook", "sportsbetting", "SportsBettingPicks",
                    "esportsbetting", "sportsbookadvice"]
