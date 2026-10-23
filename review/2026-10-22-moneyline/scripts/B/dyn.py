"""Online (dynamic) Bradley-Terry ratings on the logit scale: Elo, diagonal Kalman (Glicko-like EKF),
full-covariance EKF. A match is priced at kick-off from every result settled by then (kick-off + 36 min)."""
import common as C
import math
from collections import deque
import numpy as np

LAG = 36 * 60.0  # seconds


def settle_times(t, midnight=False):
    """When each result counts: kick-off + 36 min, or (midnight=True, the static fits' cadence) the first
    UTC midnight at or after that."""
    s = t + LAG
    if midnight:
        s = np.ceil(s / 86400.0) * 86400.0
    return s


def _arrays(d, midnight=False):
    t = d.tsec.to_numpy()
    return dict(settle=settle_times(t, midnight), t=d.tsec.to_numpy(), g1=d.g1.to_numpy(), g2=d.g2.to_numpy(), k1=d.k1.to_numpy(),
                k2=d.k2.to_numpy(), st=d.st.to_numpy(), y=d.y.to_numpy(), m=d.margin.to_numpy(),
                day=d.tday.to_numpy())


def elo(d, K=0.08, Knew=0.0, nc=20.0, h=0.05, Kh=0.0, r0=0.0, Kt=0.0, Kgt=0.0, stream_home=False,
        mov=None, tau=None, ac=0.0, A=None):
    """Elo on the logit scale. K(n) = K + Knew/(1+n/nc) with n the gamer's games so far.
    mov=(a, c): update scaled by ((|margin|+c)/(7+c))^a (a=0 off); ac: 538-style autocorrelation damping
    of the MOV multiplier by the winner's pre-match edge, 1/(1+ac*edge_winner). tau: ratings shrink toward 0
    with e-folding time tau days. Returns p_home for every row of d (kick-off order)."""
    A = A or _arrays(d)
    n = len(A["t"])
    NG, NT = C.ngamers(d), C.nteams(d)
    r = np.full(NG, r0); ng = np.zeros(NG); last = np.zeros(NG)
    T = np.zeros(NT); GT = np.zeros((NG, NT))
    hh = np.full(3, h)
    p = np.empty(n)
    q = deque()
    t, g1, g2, k1, k2, st, y, mg = A["t"], A["g1"], A["g2"], A["k1"], A["k2"], A["st"], A["y"], A["m"]
    for i in range(n):
        ti = t[i]
        while q and q[0][0] <= ti:
            _, j, zj = q.popleft()
            yj = y[j]
            if yj != yj:  # tie: no update
                continue
            a, b = g1[j], g2[j]
            pj = 1.0 / (1.0 + math.exp(-zj))
            e = yj - pj
            if mov is not None and mov[0] != 0:
                mult = ((abs(mg[j]) + mov[1]) / (7.0 + mov[1])) ** mov[0]
                if ac:
                    edge = zj if yj == 1 else -zj
                    mult /= (1.0 + ac * max(edge, 0.0))
                e *= mult
            Ka = K + Knew / (1.0 + ng[a] / nc)
            Kb = K + Knew / (1.0 + ng[b] / nc)
            r[a] += Ka * e; r[b] -= Kb * e
            ng[a] += 1; ng[b] += 1
            if Kh:
                if stream_home:
                    hh[st[j]] += Kh * e
                else:
                    hh[:] += Kh * e
            if Kt:
                T[k1[j]] += Kt * e; T[k2[j]] -= Kt * e
            if Kgt:
                GT[a, k1[j]] += Kgt * e; GT[b, k2[j]] -= Kgt * e
        a, b = g1[i], g2[i]
        if tau:
            for g in (a, b):
                if last[g]:
                    r[g] = r0 + (r[g] - r0) * math.exp(-(ti - last[g]) / 86400.0 / tau)
                last[g] = ti
        z = hh[st[i]] + r[a] - r[b] + T[k1[i]] - T[k2[i]] + GT[a, k1[i]] - GT[b, k2[i]]
        p[i] = 1.0 / (1.0 + math.exp(-z))
        q.append((A['settle'][i], i, z))
    return p


def kalman(d, v0=0.3, q_day=0.002, q_game=0.0, h=0.05, r0=0.0, vt0=0.0, qt_day=0.0, vgt0=0.0, qgt_day=0.0,
           integrate=True, upd="ekf", hv0=0.0, stream_home=False, mov=None, A=None, return_state=False,
           vp0=0.0, qp_day=0.0):
    """Diagonal EKF (Glicko-like): each gamer N(m, v); v grows q_day per day and q_game per game; teams and
    gamer x team states optional (vt0, vgt0 > 0); home h (a state with variance hv0 if hv0 > 0).
    Price p = sigmoid(z / sqrt(1 + pi s / 8)) when integrate (s = sum of the variances in play).
    Update: Laplace/EKF one step at the prior mean: m += v x (y - p0)/(1 + s I), v -= v^2 I/(1 + s I)."""
    A = A or _arrays(d)
    n = len(A["t"])
    NG, NT = C.ngamers(d), C.nteams(d)
    m = np.full(NG, r0); v = np.full(NG, v0); tl = np.full(NG, np.nan)
    mt = np.zeros(NT); vt = np.full(NT, vt0); ttl = np.full(NT, np.nan)
    mgt = np.zeros((NG, NT)); vgt = np.full((NG, NT), vgt0); gtl = np.full((NG, NT), np.nan)
    hm = np.full(3, h); hv = np.full(3, hv0)
    mp = np.zeros((NG, NG)); vp = np.full((NG, NG), vp0); ptl = np.full((NG, NG), np.nan)
    p = np.empty(n)
    q = deque()
    t, g1, g2, k1, k2, st, y, mg = A["t"], A["g1"], A["g2"], A["k1"], A["k2"], A["st"], A["y"], A["m"]
    use_t, use_gt, use_p = vt0 > 0, vgt0 > 0, vp0 > 0
    pi8 = math.pi / 8
    for i in range(n):
        ti = t[i]
        while q and q[0][0] <= ti:
            _, j, z, s, a, b, ka, kb, sj = q.popleft()
            yj = y[j]
            if yj != yj:
                continue
            p0 = 1.0 / (1.0 + math.exp(-z))
            if upd == "probit":  # moment-matching style: use the integrated probability
                p0 = 1.0 / (1.0 + math.exp(-z / math.sqrt(1 + pi8 * s)))
            I = p0 * (1 - p0)
            if mov is not None and mov[0] != 0:
                I_mult = ((abs(mg[j]) + mov[1]) / (7.0 + mov[1])) ** mov[0]
            else:
                I_mult = 1.0
            e = (yj - p0) * I_mult
            Ie = I * I_mult
            den = 1.0 + s * Ie
            m[a] += v[a] * e / den; m[b] -= v[b] * e / den
            v[a] -= v[a] ** 2 * Ie / den; v[b] -= v[b] ** 2 * Ie / den
            if use_t:
                mt[ka] += vt[ka] * e / den; mt[kb] -= vt[kb] * e / den
                vt[ka] -= vt[ka] ** 2 * Ie / den; vt[kb] -= vt[kb] ** 2 * Ie / den
            if use_gt:
                mgt[a, ka] += vgt[a, ka] * e / den; mgt[b, kb] -= vgt[b, kb] * e / den
                vgt[a, ka] -= vgt[a, ka] ** 2 * Ie / den; vgt[b, kb] -= vgt[b, kb] ** 2 * Ie / den
            if use_p:  # pair effect stored at [min, max], entering +1 when the lower id is at home
                lo, hi, sg = (a, b, 1.0) if a < b else (b, a, -1.0)
                mp[lo, hi] += sg * vp[lo, hi] * e / den
                vp[lo, hi] -= vp[lo, hi] ** 2 * Ie / den
            if hv0 > 0:
                hh = sj if stream_home else slice(None)
                hm[hh] += hv[hh] * e / den
                hv[hh] -= hv[hh] ** 2 * Ie / den
        a, b, ka, kb, sj = g1[i], g2[i], k1[i], k2[i], st[i]
        day = ti / 86400.0
        for g in (a, b):
            if tl[g] == tl[g]:
                v[g] += q_day * (day - tl[g]) + q_game
            tl[g] = day
        if use_t:
            for k in (ka, kb):
                if ttl[k] == ttl[k]:
                    vt[k] += qt_day * (day - ttl[k])
                ttl[k] = day
        if use_gt:
            for g, k in ((a, ka), (b, kb)):
                if gtl[g, k] == gtl[g, k]:
                    vgt[g, k] += qgt_day * (day - gtl[g, k])
                gtl[g, k] = day
        z = hm[sj] + m[a] - m[b]
        s = v[a] + v[b] + hv[sj]
        if use_t:
            z += mt[ka] - mt[kb]; s += vt[ka] + vt[kb]
        if use_p:
            lo, hi, sg = (a, b, 1.0) if a < b else (b, a, -1.0)
            if ptl[lo, hi] == ptl[lo, hi]:
                vp[lo, hi] += qp_day * (day - ptl[lo, hi])
            ptl[lo, hi] = day
            z += sg * mp[lo, hi]; s += vp[lo, hi]
        if use_gt:
            z += mgt[a, ka] - mgt[b, kb]; s += vgt[a, ka] + vgt[b, kb]
        zz = z / math.sqrt(1 + pi8 * s) if integrate else z
        p[i] = 1.0 / (1.0 + math.exp(-zz))
        q.append((A['settle'][i], i, z, s, a, b, ka, kb, sj))
    if return_state:
        return p, dict(m=m, v=v, mt=mt, vt=vt, h=hm)
    return p


def kalman_full(d, v0=0.3, q_day=0.002, h=0.05, hv0=0.01, hq_day=0.0, vt0=0.05, qt_day=0.0, integrate=True,
                stream_home=False, A=None):
    """Full-covariance EKF over [home x3, gamers, teams]: a gamer's and team's estimates are correlated
    through the matches they share. Drift on the diagonal per elapsed day (all states, lazily per event)."""
    A = A or _arrays(d)
    n = len(A["t"])
    NG, NT = C.ngamers(d), C.nteams(d)
    P_ = 3 + NG + NT
    th = np.zeros(P_); th[:3] = h
    P = np.zeros((P_, P_))
    P[np.arange(3), np.arange(3)] = hv0
    if not stream_home:
        P[:3, :3] = hv0
    seen = np.zeros(P_, bool); seen[:3] = True
    qd = np.zeros(P_); qd[3:3 + NG] = q_day; qd[3 + NG:] = qt_day; qd[:3] = hq_day
    init_v = np.zeros(P_); init_v[3:3 + NG] = v0; init_v[3 + NG:] = vt0
    if vt0 <= 0:
        init_v[3 + NG:] = 0
    p = np.empty(n)
    q = deque()
    t, g1, g2, k1, k2, st, y = A["t"], A["g1"], A["g2"], A["k1"], A["k2"], A["st"], A["y"]
    tlast = t[0] / 86400.0
    pi8 = math.pi / 8
    for i in range(n):
        ti = t[i]
        day = ti / 86400.0
        if day > tlast:
            dd = day - tlast
            idx = np.arange(P_)
            P[idx, idx] += qd * dd * seen
            if not stream_home and hq_day:
                P[:3, :3] += hq_day * dd - np.diag(np.full(3, hq_day * dd))
            tlast = day
        while q and q[0][0] <= ti:
            _, j, ix, xv = q.popleft()
            yj = y[j]
            if yj != yj:
                continue
            z = th[ix] @ xv
            Px = P[:, ix] @ xv
            s = xv @ Px[ix]
            p0 = 1.0 / (1.0 + math.exp(-z))
            I = p0 * (1 - p0)
            den = 1.0 + s * I
            th += Px * (yj - p0) / den
            P -= np.outer(Px, Px) * (I / den)
        a, b = 3 + g1[i], 3 + g2[i]
        ix = [st[i], a, b]
        xv = [1.0, 1.0, -1.0]
        if vt0 > 0:
            ix += [3 + NG + k1[i], 3 + NG + k2[i]]
            xv += [1.0, -1.0]
        ix = np.array(ix); xv = np.array(xv)
        for u in ix:
            if not seen[u]:
                P[u, u] = init_v[u]; seen[u] = True
        z = th[ix] @ xv
        s = xv @ P[np.ix_(ix, ix)] @ xv
        zz = z / math.sqrt(1 + pi8 * s) if integrate else z
        p[i] = 1.0 / (1.0 + math.exp(-zz))
        q.append((A['settle'][i], i, ix, xv))
    return p


def kalman2(d, v0=1.5, q_day=0.001, q_game=0.0, h=0.01, r0=0.0, vt0=0.02, qt_day=0.0002, vf=0.02, tauf=0.2,
            integrate=True, A=None, mov=None):
    """Two-timescale EKF: each gamer = slow random-walk strength m (drift q_day/day) + fast mean-reverting
    form f (Ornstein-Uhlenbeck, stationary variance vf, e-folding tauf days), 2x2 covariance per gamer;
    teams diagonal as in kalman()."""
    A = A or _arrays(d)
    n = len(A["t"])
    NG, NT = C.ngamers(d), C.nteams(d)
    m = np.full(NG, r0); f = np.zeros(NG)
    v = np.full(NG, v0); w = np.full(NG, vf); c = np.zeros(NG); tl = np.full(NG, np.nan)
    mt = np.zeros(NT); vt = np.full(NT, vt0); ttl = np.full(NT, np.nan)
    p = np.empty(n)
    q = deque()
    t, g1, g2, k1, k2, st, y, mg = A["t"], A["g1"], A["g2"], A["k1"], A["k2"], A["st"], A["y"], A["m"]
    use_t = vt0 > 0
    pi8 = math.pi / 8
    for i in range(n):
        ti = t[i]
        while q and q[0][0] <= ti:
            _, j, z, s, a, b, ka, kb = q.popleft()
            yj = y[j]
            if yj != yj:
                continue
            p0 = 1.0 / (1.0 + math.exp(-z))
            I = p0 * (1 - p0)
            mult = ((abs(mg[j]) + mov[1]) / (7.0 + mov[1])) ** mov[0] if (mov is not None and mov[0]) else 1.0
            e = (yj - p0) * mult
            Ie = I * mult
            den = 1.0 + s * Ie
            for g, sg in ((a, 1.0), (b, -1.0)):
                ka_ = v[g] + c[g]; kf_ = c[g] + w[g]
                m[g] += sg * ka_ * e / den; f[g] += sg * kf_ * e / den
                v[g] -= ka_ * ka_ * Ie / den; w[g] -= kf_ * kf_ * Ie / den; c[g] -= ka_ * kf_ * Ie / den
            if use_t:
                mt[ka] += vt[ka] * e / den; mt[kb] -= vt[kb] * e / den
                vt[ka] -= vt[ka] ** 2 * Ie / den; vt[kb] -= vt[kb] ** 2 * Ie / den
        a, b, ka, kb, sj = g1[i], g2[i], k1[i], k2[i], st[i]
        day = ti / 86400.0
        for g in (a, b):
            if tl[g] == tl[g]:
                dt = day - tl[g]
                rho = math.exp(-dt / tauf)
                v[g] += q_day * dt + q_game
                f[g] *= rho; c[g] *= rho; w[g] = rho * rho * w[g] + vf * (1 - rho * rho)
            tl[g] = day
        if use_t:
            for k in (ka, kb):
                if ttl[k] == ttl[k]:
                    vt[k] += qt_day * (day - ttl[k])
                ttl[k] = day
        z = h + m[a] + f[a] - m[b] - f[b]
        s = v[a] + w[a] + 2 * c[a] + v[b] + w[b] + 2 * c[b]
        if use_t:
            z += mt[ka] - mt[kb]; s += vt[ka] + vt[kb]
        zz = z / math.sqrt(1 + pi8 * s) if integrate else z
        p[i] = 1.0 / (1.0 + math.exp(-zz))
        q.append((A['settle'][i], i, z, s, a, b, ka, kb))
    return p


def kalman3(d, v0=3.5, q_day=0.002, q_game=0.0, h=0.01, r0=0.0, vt0=10.0, qt_day=1e-5, vf=0.05, tauf=0.3,
            hv0=0.01, vp0=0.0, qp_day=0.0, integrate=True, A=None, mov=None):
    """kalman2 (slow strength + fast OU form per gamer, teams) with stream-specific home states (variance hv0)
    and head-to-head pair effects (vp0, drift qp_day)."""
    A = A or _arrays(d)
    n = len(A["t"])
    NG, NT = C.ngamers(d), C.nteams(d)
    m = np.full(NG, r0); f = np.zeros(NG)
    v = np.full(NG, v0); w = np.full(NG, vf); c = np.zeros(NG); tl = np.full(NG, np.nan)
    mt = np.zeros(NT); vt = np.full(NT, vt0); ttl = np.full(NT, np.nan)
    hm = np.full(3, h); hv = np.full(3, hv0)
    mp = np.zeros((NG, NG)); vp = np.full((NG, NG), vp0); ptl = np.full((NG, NG), np.nan)
    p = np.empty(n)
    q = deque()
    t, g1, g2, k1, k2, st, y, mg = A["t"], A["g1"], A["g2"], A["k1"], A["k2"], A["st"], A["y"], A["m"]
    use_p = vp0 > 0
    pi8 = math.pi / 8
    for i in range(n):
        ti = t[i]
        while q and q[0][0] <= ti:
            _, j, z, s, a, b, ka, kb, sj, lo, hi = q.popleft()
            yj = y[j]
            if yj != yj:
                continue
            p0 = 1.0 / (1.0 + math.exp(-z))
            I = p0 * (1 - p0)
            mult = ((abs(mg[j]) + mov[1]) / (7.0 + mov[1])) ** mov[0] if (mov is not None and mov[0]) else 1.0
            e = (yj - p0) * mult
            Ie = I * mult
            den = 1.0 + s * Ie
            for g, sg in ((a, 1.0), (b, -1.0)):
                ka_ = v[g] + c[g]; kf_ = c[g] + w[g]
                m[g] += sg * ka_ * e / den; f[g] += sg * kf_ * e / den
                v[g] -= ka_ * ka_ * Ie / den; w[g] -= kf_ * kf_ * Ie / den; c[g] -= ka_ * kf_ * Ie / den
            mt[ka] += vt[ka] * e / den; mt[kb] -= vt[kb] * e / den
            vt[ka] -= vt[ka] ** 2 * Ie / den; vt[kb] -= vt[kb] ** 2 * Ie / den
            hm[sj] += hv[sj] * e / den; hv[sj] -= hv[sj] ** 2 * Ie / den
            if use_p:
                sg = 1.0 if a < b else -1.0
                mp[lo, hi] += sg * vp[lo, hi] * e / den
                vp[lo, hi] -= vp[lo, hi] ** 2 * Ie / den
        a, b, ka, kb, sj = g1[i], g2[i], k1[i], k2[i], st[i]
        day = ti / 86400.0
        for g in (a, b):
            if tl[g] == tl[g]:
                dt = day - tl[g]
                rho = math.exp(-dt / tauf)
                v[g] += q_day * dt + q_game
                f[g] *= rho; c[g] *= rho; w[g] = rho * rho * w[g] + vf * (1 - rho * rho)
            tl[g] = day
        for k in (ka, kb):
            if ttl[k] == ttl[k]:
                vt[k] += qt_day * (day - ttl[k])
            ttl[k] = day
        z = hm[sj] + m[a] + f[a] - m[b] - f[b] + mt[ka] - mt[kb]
        s = v[a] + w[a] + 2 * c[a] + v[b] + w[b] + 2 * c[b] + vt[ka] + vt[kb] + hv[sj]
        lo, hi = (a, b) if a < b else (b, a)
        if use_p:
            if ptl[lo, hi] == ptl[lo, hi]:
                vp[lo, hi] += qp_day * (day - ptl[lo, hi])
            ptl[lo, hi] = day
            z += (1.0 if a < b else -1.0) * mp[lo, hi]; s += vp[lo, hi]
        zz = z / math.sqrt(1 + pi8 * s) if integrate else z
        p[i] = 1.0 / (1.0 + math.exp(-zz))
        q.append((A['settle'][i], i, z, s, a, b, ka, kb, sj, lo, hi))
    return p
