import os, sys, numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import markets, stream
# 1. prob_above/below with pushes and complementarity
pmf = np.zeros(161); pmf[40] = 0.3; pmf[41] = 0.2; pmf[44] = 0.5
for line in (40.0, 40.5, 41.0, 44.0):
    a, b = markets.prob_above(pmf, 0, line), markets.prob_below(pmf, 0, line)
    print("line", line, "over", float(a), "under", float(b), "sum", float(a + b))
# 2. even_line / key_line on a spiky margin distribution
m = np.zeros(201); off = markets.MARGIN_MAX
for k, p in {-7: .1, -3: .15, 0: 0.0, 1: .05, 3: .25, 4: .05, 7: .2, 10: .1, 14: .1}.items():
    m[k + off] = p
m /= m.sum()
print("even", markets.even_line(m, off), "key", markets.key_line(m, off))
# 3. market_prob for spread home and away at the stream's own line, add to 1?
x = markets.key_line(m, off)
p52 = markets.market_prob(52, x, m, pmf); p53 = markets.market_prob(53, -x, m, pmf)
print("p52", float(p52), "p53", float(p53), "sum", float(p52 + p53))
# 4. even line where cdf hits exactly 0.5 on an integer
m2 = np.zeros(201); m2[off + 3] = 0.5; m2[off - 3] = 0.5
print("even line symmetric", markets.even_line(m2, off), "P(above)", float(markets.prob_above(m2, off, markets.even_line(m2, off))))
