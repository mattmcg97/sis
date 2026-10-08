// Reddit eFootball/eSoccer/eBasketball betting monitor - browser console version.
// Usage: open https://www.reddit.com in Chrome/Edge, press F12 -> Console,
// paste this whole file, press Enter. A CSV downloads when it finishes.
// Generated from config.py - edit CFG below or re-generate.
(async () => {
const CFG = {"PLAYERS": ["KRAKEN", "SABRE", "RELIC", "JUDGEMENT", "ROCKET", "PHOENIX", "METEOR", "SAGE", "DREAD", "MERLIN", "RICOCHET", "CLOCKWORK", "FUSE", "NIGHTMARE", "ENFORCER", "KILLJOY", "KATANA", "HERMES", "ACE", "ROMAN", "ORIGINAL", "REGRET", "MAVERICK", "STEEL", "SPARK", "MINDFLAYER", "GADGET", "IMPERIAL", "SENTRY", "HURRICANE", "PINNACLE", "KEVLAR", "MERCENARY", "PRISMATIC", "SCHOLAR", "PURSUIT", "MIRAGE", "SHARK", "CLASH", "STRIDER", "LEGEND", "WARRIOR", "OFFICER", "HEADHUNTER", "MISSILE", "MEMENTO", "PRECISION", "SHROUD", "LUNA", "SHURIKEN", "LIZARD", "VALIANT", "MERCURY", "FENRIR", "MIMIC", "SUMMIT", "LYNX", "CRYSTAL", "PSYCHIC", "MONSTER", "MIDAS", "UNICORN", "LEO", "DESTINY", "GENIUS", "ASTRONAUT", "LAZARUS", "OVERDRIVE", "PATHFINDER", "OCTANE", "DON", "NIRVANA", "AVATAR", "SENTINEL", "TOMBSTONE", "SORCERER", "ANVIL", "DAME", "SEEKER", "BLAZE", "NEON", "SPLINTER", "AERO", "INVENTOR", "MARVEL", "EMBER", "PARADOX", "FRACTURE", "IDOL"], "SPORT_TERMS": ["efootball", "e-football", "esoccer", "e-soccer", "ebasketball", "e-basketball", "enfl", "e-nfl", "madden sim", "esports football", "virtual football", "cyber football", "cyber basketball", "fifa 8 min", "2k sim", "nba 2k esports"], "BET_TERMS": {"under": 3, "unders": 3, "total": 3, "totals": 3, "o/u": 3, "over/under": 3, "total points": 3, "points total": 3, "moneyline": 2, "ml": 2, "spread": 2, "ats": 2, "handicap": 2, "tip": 1, "tips": 1, "pick": 1, "picks": 1, "lock": 1, "parlay": 1, "units": 1, "telegram": 2, "discord": 1, "vip": 2, "fixed": 3, "sure bet": 2}, "BOOKS": ["fanduel", "fan duel", "hard rock", "hardrock", "hard rock bet", "draftkings", "bet365", "betano", "pinnacle", "betmgm"], "REGIONS": ["ontario", "brazil", "brasil", "belgium", "belgie", "belgique", "usa", "florida", "new jersey"], "QUERIES": ["efootball (bet OR betting OR tips OR picks OR under OR total)", "esoccer (bet OR betting OR tips OR picks OR under OR total)", "ebasketball (bet OR betting OR tips OR picks OR under OR total)", "\"e-football\" betting", "\"e-soccer\" betting", "\"e-basketball\" betting", "\"cyber football\" bet", "\"cyber basketball\" bet", "esports (fanduel OR \"hard rock\") (under OR total OR moneyline OR spread)", "efootball fanduel", "efootball \"hard rock\"", "esoccer telegram tips", "ebasketball telegram tips", "esoccer apostas", "ebasketball apostas"], "SUBREDDIT_QUERIES": ["esoccer", "ebasketball", "efootball", "esports betting", "esoccer betting", "ebasketball betting", "cyber football", "fifa betting", "betting tips", "apostas esports", "sports picks"], "WATCH_SUBREDDITS": ["sportsbook", "sportsbetting", "SportsBettingPicks", "esportsbetting", "sportsbookadvice"]};
const TIME = "year";      // day | week | month | year | all
const PER_QUERY = 250;    // max results per query
const DELAY_MS = 3000;    // pause between requests

const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const has = (text, terms) => terms.filter(t => new RegExp("(?<!\\w)" + esc(t) + "(?!\\w)").test(text));
const capsRe = new RegExp("\\b(" + CFG.PLAYERS.join("|") + ")\\b", "g");
const anyRe = new RegExp("\\b(" + CFG.PLAYERS.join("|") + ")\\b", "gi");
const BET = Object.keys(CFG.BET_TERMS);

function score(text) {
  const lower = text.toLowerCase();
  const sports = has(lower, CFG.SPORT_TERMS);
  if (!sports.length) return [0, null];
  const bets = has(lower, BET), books = has(lower, CFG.BOOKS), regions = has(lower, CFG.REGIONS);
  const players = new Set(text.match(capsRe) || []);
  if (bets.length || books.length) (text.match(anyRe) || []).forEach(p => players.add(p.toUpperCase()));
  const s = 2 + bets.reduce((a, b) => a + CFG.BET_TERMS[b], 0) + 2 * books.length + 3 * players.size + regions.length;
  return [s, {sports, bets, books, regions, players: [...players].sort()}];
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function get(path, params) {
  const url = path + "?" + new URLSearchParams({...params, raw_json: 1});
  for (let i = 0; i < 5; i++) {
    await sleep(DELAY_MS);
    const r = await fetch(url, {credentials: "include"});
    if (r.status === 429) { console.log("rate limited, waiting 60s"); await sleep(60000); continue; }
    if (!r.ok) { console.warn(r.status, url); return null; }
    return r.json();
  }
  return null;
}
async function* listing(path, params, limit) {
  let after = null, got = 0;
  while (got < limit) {
    const j = await get(path, {...params, limit: 100, ...(after ? {after} : {})});
    if (!j) return;
    for (const c of j.data.children) { yield c.data; got++; }
    after = j.data.after;
    if (!after) return;
  }
}

const searches = CFG.QUERIES.map(q => ["/search.json", {q, sort: "new", t: TIME, type: "link"}]);
for (const sub of CFG.WATCH_SUBREDDITS)
  for (const q of ["efootball", "esoccer", "ebasketball", "cyber"])
    searches.push([`/r/${sub}/search.json`, {q, sort: "new", t: TIME, restrict_sr: 1}]);

const seen = new Set(), rows = [];
for (const [i, [path, params]] of searches.entries()) {
  console.log(`query ${i + 1}/${searches.length}: ${path} ${params.q}`);
  for await (const p of listing(path, params, PER_QUERY)) {
    if (seen.has(p.id)) continue;
    seen.add(p.id);
    const [s, d] = score(`${p.title}\n${p.selftext || ""}`);
    if (!s) continue;
    console.log(`  hit [${s}] r/${p.subreddit}: ${p.title.slice(0, 70)}`);
    rows.push({score: s, subreddit: p.subreddit, author: p.author, title: p.title,
      created: new Date(p.created_utc * 1000).toISOString(), url: "https://reddit.com" + p.permalink,
      ...Object.fromEntries(Object.entries(d).map(([k, v]) => [k, v.join("; ")]))});
  }
}

const subs = {};
for (const q of CFG.SUBREDDIT_QUERIES) {
  console.log("subreddit search:", q);
  for await (const s of listing("/subreddits/search.json", {q}, 100)) {
    const [sc] = score(`${s.display_name} ${s.title || ""} ${s.public_description || ""} tips`);
    if (sc && !subs[s.display_name])
      subs[s.display_name] = {score: sc, subreddit: s.display_name, author: `${s.subscribers} members`,
        title: "[CANDIDATE SUBREDDIT] " + (s.public_description || "").slice(0, 150),
        url: "https://reddit.com/r/" + s.display_name};
  }
}

rows.sort((a, b) => b.score - a.score);
const all = [...rows, ...Object.values(subs).sort((a, b) => b.score - a.score)];
const cols = ["score", "subreddit", "author", "title", "created", "url", "sports", "bets", "books", "regions", "players"];
const q = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
const csv = "\ufeff" + [cols.join(","), ...all.map(r => cols.map(c => q(r[c])).join(","))].join("\r\n");
const a = document.createElement("a");
a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
a.download = "reddit_hits.csv"; a.click();
console.log(`Done: ${rows.length} posts (of ${seen.size} scanned), ${Object.keys(subs).length} candidate subreddits.`);
})();
