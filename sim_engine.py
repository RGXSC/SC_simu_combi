"""
Supply Chain Simulation Engine v3 — standalone module.
No Streamlit dependency. Safe for multiprocessing on all platforms.
"""
import math


def sim_fast(weeks, init_sa, init_sb, init_cw, init_semi, init_rm,
             mat_lt, semi_lt, fp_lt, dist_lt, order_freq,
             cap_start, cap_ramp, base_forecast, demand_mult, smart_distrib,
             demand_pct_a=50):
    """Ultra-lean simulation.
    demand_pct_a: % of total demand going to Store A (0-100). Default 50.
    Returns (total_sales, total_missed, total_demand,
             total_produced, stockout_weeks, missed_a, missed_b)"""
    _ceil = math.ceil; _min = min; _max = max; _round = round
    pct_a = demand_pct_a / 100.0
    ramp_end = 4 if demand_mult > 1.0 else 1
    ml = max(1, mat_lt); sl = max(1, semi_lt); fl = max(1, fp_lt); dl = max(1, dist_lt)
    mp = [0]*ml; sp = [0]*sl; fpp = [0]*fl; dpa = [0]*dl; dpb = [0]*dl
    i0=0; i1=0; i2=0; i3=0; i4=0
    sa = float(init_sa); sb = float(init_sb)
    rm = float(init_rm); sm = float(init_semi); cw = float(init_cw)
    pb = 0.0; scap = cap_start+0.0; smcap = cap_start+0.0; fcap = cap_start+0.0
    sa_ = False; sma_ = False; fa_ = False
    co = 0.0; cas = 0.0; cov = mat_lt + semi_lt + fp_lt + dist_lt + order_freq
    ts = 0.0; tm_ = 0.0; td = 0.0; tp = 0.0; so = 0; ma_ = 0.0; mb_ = 0.0
    bf = base_forecast; cs = float(cap_start); cr = cap_ramp; mx = cs * 10
    dem_full = int(bf * demand_mult)
    for w in range(1, weeks+1):
        if demand_mult <= 1.0:
            dem = dem_full
        elif w >= ramp_end:
            dem = dem_full
        else:
            p_ = (w-1)/(ramp_end-1) if ramp_end > 1 else 1.0
            dem = int(bf + (dem_full - bf) * p_)
        da = _round(dem * pct_a); db = dem - da
        mar = mp[i0]; sar = sp[i1]; far = fpp[i2]; dar = dpa[i3]; dbr = dpb[i4]
        cas += dar + dbr
        ava = sa + dar
        if da <= ava: s_a = da; m_a = 0
        else: s_a = ava; m_a = da - ava
        sa = ava - s_a
        avb = sb + dbr
        if db <= avb: s_b = db; m_b = 0
        else: s_b = avb; m_b = db - avb
        sb = avb - s_b
        ts += s_a + s_b; tm_ += m_a + m_b; td += dem; ma_ += m_a; mb_ += m_b
        if m_a + m_b > 0.5: so += 1
        if pb > 0.01:
            if not sa_: sa_ = True; scap = cs
            else: scap = _min(scap*(1+cr), mx)
            sh = _ceil(_min(pb, scap)); pb -= sh
        else: sh = 0; sa_ = False; scap = cs
        rm += mar; sm += sar; cw += far
        if rm > 0.01:
            if not sma_: sma_ = True; smcap = cs
            else: smcap = _min(smcap*(1+cr), mx)
            si = _ceil(_min(rm, smcap)); rm -= si
        else: si = 0; sma_ = False; smcap = cs
        if sm > 0.01:
            if not fa_: fa_ = True; fcap = cs
            else: fcap = _min(fcap*(1+cr), mx)
            fi = _ceil(_min(sm, fcap)); sm -= fi
        else: fi = 0; fa_ = False; fcap = cs
        tp += fi
        ship = _ceil(cw) if cw > 0.01 else 0; cw = 0.0
        if smart_distrib and ship > 0:
            na = _max(0, da*dl - sa); nb = _max(0, db*dl - sb); tn = na + nb
            if tn > 0.01: aa = _round(ship*na/tn); ab = ship - aa
            else: aa = _round(ship * pct_a); ab = ship - aa
        else:
            aa = _round(ship * 0.5); ab = ship - aa
        mp[i0] = sh; sp[i1] = si; fpp[i2] = fi; dpa[i3] = aa; dpb[i4] = ab
        i0 = (i0+1)%ml; i1 = (i1+1)%sl; i2 = (i2+1)%fl; i3 = (i3+1)%dl; i4 = (i4+1)%dl
        if order_freq == 1 or w % order_freq == 0:
            st_ = sa + sb; pnd = co - cas; tgt = dem * cov
            gap = tgt - st_ - pnd
            if gap > 0: od = _ceil(gap); co += od; pb += od
    return ts, tm_, td, tp, so, ma_, mb_


def worker_chunk(jobs):
    """Process a chunk of simulation jobs. Job tuple:
    (dm, wks, pf, cr, sm_d, mat_lt, semi_lt, fp_lt, dist_lt, total_lt,
     stock_idx, sp_, wp_, sep_, rp_, total_stock, cap_start, base_forecast, demand_pct_a)
    Returns tuples with 24 columns (added demand_pct_a)."""
    results = []
    _sim = sim_fast
    for job in jobs:
        (dm, wks, pf, cr, sm_d, mat_lt, semi_lt, fp_lt, dist_lt, total_lt,
         stock_idx, sp_, wp_, sep_, rp_, total_stock, cap_start, base_forecast,
         demand_pct_a) = job
        init_store = int(total_stock * sp_ / 100)
        init_cw = int(total_stock * wp_ / 100)
        init_semi = int(total_stock * sep_ / 100)
        init_rm = total_stock - init_store - init_cw - init_semi
        # Initial store stock: always 50/50 (deliberate — push baseline)
        init_sa = init_store // 2
        init_sb = init_store - init_sa
        ts, tm, td, tp, so, ma, mb = _sim(
            wks, init_sa, init_sb, init_cw, init_semi, init_rm,
            mat_lt, semi_lt, fp_lt, dist_lt, pf,
            cap_start, cr, base_forecast, dm, sm_d, demand_pct_a
        )
        results.append((
            dm, wks, pf, cr, int(sm_d),
            mat_lt, semi_lt, fp_lt, dist_lt, total_lt,
            stock_idx, sp_, wp_, sep_, rp_, total_stock,
            ts, tm, td, tp, so, ma, mb, demand_pct_a
        ))
    return results
