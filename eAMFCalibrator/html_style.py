"""The one stylesheet, shared by every HTML view.

Pulled out of html_full so the in-drive report cannot drift from
the main one: the design tokens, the gap ramp and the table rules
are defined once and embedded verbatim by both.
"""

CSS = r'''<style>
  :root {
    --bg:#f7f7f5; --panel:#fff; --ink:#1a1a18; --dim:#6b6b66; --line:#e2e2dd;
    --g0:#007743; --g1:#726400; --g2:#8f5300; --g3:#a73c00; --g4:#b3261e;
    --good:#1c7c4a; --bad:#b3261e; --warn:#8a6d1f; --accent:#2d4a7c;
    --head:#f0f0ec; --axis:#eaeef4; --pick:#fdf0c8; --hover:#f2f2ef;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#17171a; --panel:#1f1f23; --ink:#ededea; --dim:#9a9a95; --line:#32323a;
      --g0:#22b970; --g1:#b29d00; --g2:#dd8400; --g3:#ec7544; --g4:#ef6f66;
      --good:#4cc281; --bad:#ef6f66; --warn:#d9b451; --accent:#8fb0e8;
      --head:#26262c; --axis:#232833; --pick:#4a3f1c; --hover:#26262c;
    }
  }
  :root[data-theme="dark"] {
    --bg:#17171a; --panel:#1f1f23; --ink:#ededea; --dim:#9a9a95; --line:#32323a;
    --g0:#22b970; --g1:#b29d00; --g2:#dd8400; --g3:#ec7544; --g4:#ef6f66;
    --good:#4cc281; --bad:#ef6f66; --warn:#d9b451; --accent:#8fb0e8;
    --head:#26262c; --axis:#232833; --pick:#4a3f1c; --hover:#26262c;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);padding:16px;
       font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  .wrap{max-width:1500px;margin:0 auto}
  header{display:flex;flex-wrap:wrap;gap:4px 18px;align-items:baseline;margin-bottom:10px}
  h1{font-size:18px;margin:0;letter-spacing:-0.01em}
  h2{font-size:14px;margin:0 0 3px;letter-spacing:-0.005em}
  h3{font-size:12px;margin:14px 0 5px;color:var(--dim);text-transform:uppercase;
      letter-spacing:0.05em}
  .meta{color:var(--dim);font-size:11.5px}
  .meta b{color:var(--ink);font-weight:600}
  nav{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px;font-size:11.5px}
  nav a{color:var(--accent);text-decoration:none}
  nav a:hover{text-decoration:underline}
  .verdict{padding:9px 12px;border-radius:7px;margin-bottom:12px;background:var(--panel);
            border-left:3px solid var(--warn)}
  .verdict.good{border-left-color:var(--good)}
  .verdict.bad{border-left-color:var(--bad)}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:7px;
          padding:12px 14px;margin-bottom:12px}
  /* Foldable panels. The heading is the handle, so it gets a caret and a
     pointer; the caret turns rather than the heading moving, which keeps
     the row from jumping as sections open and shut. */
  .panel.foldable > h2{cursor:pointer;user-select:none}
  .panel.foldable > h2::before{content:"\25BE";display:inline-block;
        width:1em;color:var(--dim);transition:transform .12s ease}
  .panel.folded > h2::before{transform:rotate(-90deg)}
  .panel.folded > h2{margin-bottom:0}
  .panel.folded > .fold-body{display:none}
  .panel.foldable > h2:focus-visible{outline:2px solid var(--accent);
        outline-offset:2px;border-radius:3px}
  button.fold{font:inherit;font-size:11.5px;color:var(--accent);background:none;
        border:0;padding:0;cursor:pointer;text-decoration:underline}
  /* A native <details> gets the same handle as a folded section, so the
     two kinds of panel do not look like two kinds of thing. */
  details.panel > summary{cursor:pointer;user-select:none;font-size:14px;
        font-weight:600;letter-spacing:-0.005em;list-style:none}
  details.panel > summary::-webkit-details-marker{display:none}
  details.panel > summary::before{content:"\25BE";display:inline-block;
        width:1em;color:var(--dim);transform:rotate(-90deg);
        transition:transform .12s ease}
  details.panel[open] > summary::before{transform:none}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media (max-width:820px){.cols{grid-template-columns:1fr}}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:4px 6px;border-bottom:1px solid var(--line);
         white-space:nowrap}
  td.wrap,th.wrap{white-space:normal;text-align:left;max-width:620px}
  thead th{color:var(--dim);font-weight:500;font-size:10.5px;text-transform:uppercase;
            letter-spacing:0.04em;background:var(--head)}
  tbody th,tfoot th{text-align:left;font-weight:600}
  thead th:first-child{text-align:left}
  tbody td:first-child,tbody th:first-child{text-align:left}
  tr.subtotal td,tr.subtotal th{border-bottom:2px solid var(--line);font-size:11px}
  .good{color:var(--good);font-weight:600}
  .bad{color:var(--bad);font-weight:600}
  .warn{color:var(--warn)}
  /* Gap columns: green near zero, red far from it, on the NUMBER rather
     than behind it -- a wash of filled cells reads as a heat map when
     what is wanted is a table.
     HUE ONLY. Every step sits at the same OKLCH lightness as --bad in
     its own theme (0.501 light, 0.693 dark) and differs from its
     neighbours by hue alone, so the ramp sweeps green -> amber -> orange
     -> red without ever getting brighter, and its last step IS --bad:
     #b3261e and #ef6f66, the same red the Brier columns already use.
     The cost is named rather than hidden: green and red at equal
     lightness are the one pair red-green colour blindness cannot
     separate, and a constant-lightness ramp gives up the second channel
     that used to carry the ordering. What carries it instead is the
     number, which is in the cell and is the real answer, and the key
     above each table, which names every band. Every step still clears
     4.5:1 against its panel -- 5.6 at worst -- so it is readable type
     first and coloured second. */
  td.g0{color:var(--g0);font-weight:600}
  td.g1{color:var(--g1);font-weight:600}
  td.g2{color:var(--g2);font-weight:600}
  td.g3{color:var(--g3);font-weight:600}
  td.g4{color:var(--g4);font-weight:600}
           font-size:10.5px;font-variant-numeric:tabular-nums}
                 margin-right:5px}
        font-weight:600}
  .dim{color:var(--dim);font-weight:400}
  .stats{display:flex;flex-wrap:wrap;gap:6px 20px;margin:0}
  .stats dt{color:var(--dim);font-size:10.5px;text-transform:uppercase;
             letter-spacing:0.04em}
  .stats dd{margin:1px 0 0;font-size:13px;font-variant-numeric:tabular-nums}
  td.ax,th.ax{background:var(--axis);text-align:left}
  thead th.ax{color:var(--ink)}
  .tag{color:var(--dim);font-weight:400;font-size:11px;letter-spacing:0}
  .count{color:var(--dim);font-size:11px;margin:0 0 8px}
  .filter{display:flex;gap:10px;align-items:center;margin:0 0 8px}
  .filter .count{margin:0}
  .filter input{font:inherit;font-size:12px;padding:4px 8px;min-width:220px;
                 color:var(--ink);background:var(--bg);border:1px solid var(--line);
                 border-radius:5px}
  .filter input:focus{outline:2px solid var(--accent);outline-offset:-1px}
  th[title]{cursor:help;border-bottom:1px dotted var(--dim)}
  tbody tr:hover > *{background:var(--hover)}
  /* Click a row to pin it. Inset shadows rather than a border, so pinning
     never reflows the table. */
  tbody tr.picked > *{background:var(--pick);font-weight:600}
  tbody tr.picked > :first-child{box-shadow:inset 4px 0 0 var(--warn)}
  tbody tr.picked > :last-child{box-shadow:inset -4px 0 0 var(--warn)}
  tbody tr.thin.picked > *{opacity:1}
  tbody tr.notlive > *{color:var(--dim);font-style:italic}
  tbody tr.notlive.picked > *{color:var(--ink);font-style:normal}
  tbody tr.picked .dim{color:var(--ink)}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
  details.panel{padding:0}
  details.panel > summary{cursor:pointer;padding:12px 14px;font-size:14px;
                           font-weight:700;letter-spacing:-0.005em;list-style:none}
  /* each candidate's group of columns starts with a rule */
  th.grp,td.grp{border-left:2px solid var(--line)}
  details.panel > summary::-webkit-details-marker{display:none}
  details.panel > summary::before{content:"\25B8 ";color:var(--dim)}
  details.panel[open] > summary::before{content:"\25BE "}
  details.panel[open] > summary{border-bottom:1px solid var(--line)}
  details.panel .panel{border:none;margin:0;border-bottom:1px solid var(--line);
                        border-radius:0}
  tr.thin td,tr.thin th{opacity:0.45}
  .scroll{max-height:70vh;overflow:auto;border:1px solid var(--line);border-radius:5px}
  .scroll thead th{position:sticky;top:0;z-index:1}
  .sortable thead th{cursor:pointer;user-select:none}
  .sortable thead th:hover{color:var(--ink)}
  .sortable thead th.asc::after{content:" \\2191"}
  .sortable thead th.desc::after{content:" \\2193"}
</style>'''
