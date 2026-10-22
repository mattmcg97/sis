const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({ executablePath: undefined });
  const page = await browser.newPage();
  page.on('console', m => console.log('console:', m.text()));
  await page.goto('http://127.0.0.1:18751/');
  await page.waitForFunction(() => typeof META !== 'undefined' && META !== null);
  const notBusy = () => page.waitForFunction(() => !document.querySelector('main').classList.contains('busy'), null, {timeout: 120000});
  await page.fill('#match', 'AF000');
  await page.dispatchEvent('#match', 'change');
  await page.waitForFunction(() => MATCH_LOADED === 'AF000', null, {timeout: 60000});
  await notBusy();
  let t0 = Date.now();
  await page.click('#start');
  await page.waitForFunction(() => HIST.length === 1, null, {timeout: 120000});
  await notBusy();
  console.log('start took', Date.now() - t0, 'ms');
  await page.click('button[data-a="kick"]');
  await page.waitForFunction(() => HIST.length === 2, null, {timeout: 120000});
  await notBusy();
  // fast-forward to 3 seconds left in Q1
  await page.fill('#ff', '237');
  await page.click('#ffbtn');
  await page.waitForFunction(() => HIST.length === 3, null, {timeout: 120000});
  await notBusy();
  console.log('state before:', JSON.stringify(await page.evaluate(() => { const g = HIST[HIST.length-1].game; return {period: g.period, clock: g.clock, field: g.field, down: g.down}; })));
  // run the clock and enter a play a moment later
  await page.click('#run');
  await page.waitForTimeout(300);
  await page.fill('#p_yard', '40');
  t0 = Date.now();
  await page.click('button[data-a="play"]', {force: true});
  await page.waitForTimeout(12000);
  await notBusy();
  const hist = await page.evaluate(() => HIST.map(h => ({what: h.what, period: h.game.period, clock: h.game.clock, field: h.game.field, down: h.game.down, side: h.game.side})));
  console.log('HIST:', JSON.stringify(hist, null, 0));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
