"""
Supply Chain Agility Simulator
Combinatorial batch analysis: Push vs Agile, weighted demand, visual results.
Deploy on Streamlit Cloud or run locally.
"""
import streamlit as st
import numpy as np
import pandas as pd
import time, os, sys, itertools, warnings, base64, io
warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sim_engine import sim_fast, worker_chunk

st.set_page_config(layout="wide", page_title="SC Agility Simulator", page_icon="📊")

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def generate_stock_distribs(store_vals, wh_vals, semi_vals):
    combos = set()
    for sp in store_vals:
        for wp in wh_vals:
            for sep in semi_vals:
                rp = 100 - sp - wp - sep
                if rp >= 0: combos.add((sp, wp, sep, rp))
    return sorted(combos, key=lambda x: (-x[0], -x[1], -x[2]))

def compute_lognormal_weights(demand_mults, sigma=0.5, center=1.0):
    mu = np.log(max(center, 0.01)) + sigma ** 2
    pdf_vals = stats.lognorm.pdf(demand_mults, s=sigma, scale=np.exp(mu))
    total = pdf_vals.sum()
    return (pdf_vals / total).tolist() if total > 0 else [1/len(demand_mults)]*len(demand_mults)

def weighted_mean(df, col, wc='demand_weight'):
    w = df[wc]; s = w.sum()
    return (df[col]*w).sum()/s if s > 0 else df[col].mean()

PHYS_COLS = [
    'demand_level','sim_weeks','planning_freq_wks','capacity_ramp_pct','smart_allocation',
    'lt_raw_material','lt_semifinished','lt_finished_product','lt_distribution','lt_total_weeks',
    'stock_index','pct_in_stores','pct_in_warehouse','pct_in_semifinished','pct_in_raw_material',
    'initial_stock_total','units_sold','units_missed','units_demanded','units_produced',
    'weeks_with_stockout','units_missed_store_a','units_missed_store_b','demand_split_store_a_pct'
]
ML_FEATURES = [
    'demand_level','sim_weeks','planning_freq_wks','capacity_ramp_pct','smart_allocation',
    'lt_raw_material','lt_semifinished','lt_finished_product','lt_distribution','lt_total_weeks',
    'initial_stock_total','pct_in_stores','pct_in_warehouse','pct_in_semifinished',
    'pct_in_raw_material','fixed_cost_pct','demand_split_store_a_pct'
]
PRETTY = {
    'demand_level':'Demand Level (×forecast)','sim_weeks':'Simulation Period (weeks)',
    'planning_freq_wks':'Planning Frequency (weeks)','capacity_ramp_pct':'Capacity Ramp (%/week)',
    'smart_allocation':'Smart Allocation (0/1)','lt_raw_material':'LT Raw Material (wks)',
    'lt_semifinished':'LT Semi-Finished (wks)','lt_finished_product':'LT Finished Product (wks)',
    'lt_distribution':'LT Distribution (wks)','lt_total_weeks':'LT Total (wks)',
    'initial_stock_total':'Initial Stock (units)','pct_in_stores':'% Stock in Stores',
    'pct_in_warehouse':'% Stock in Warehouse','pct_in_semifinished':'% Stock in Semi-Finished',
    'pct_in_raw_material':'% Stock in Raw Material','fixed_cost_pct':'Fixed Cost (%)',
    'demand_split_store_a_pct':'Demand Split Store A (%)',
}

# ════════════════════════════════════════════════════════════════
# BATCH ENGINE
# ════════════════════════════════════════════════════════════════
def generate_jobs(params):
    jobs = []
    for dm in params['demand_mults']:
        for wks in params['sim_weeks']:
            for pf in params['plan_freqs']:
                for cr in params['cap_ramps']:
                    for sm_d in params['smart_opts']:
                        for lt in params['lt_combos']:
                            m,s,f,d = lt
                            for stk in params['stock_levels']:
                                for sd in params['stock_distribs']:
                                    sp_,wp_,sep_,rp_ = sd
                                    for dpa in params['demand_splits']:
                                        jobs.append((dm,wks,pf,cr,sm_d,m,s,f,d,sum(lt),
                                            stk/1000.0,sp_,wp_,sep_,rp_,stk,
                                            params['cap_start'],params['base_forecast'],dpa))
    return jobs

def run_batch(params, progress_bar, status_text):
    all_jobs = generate_jobs(params)
    total = len(all_jobs)
    status_text.text(f"⏳ {total:,.0f} jobs...")
    cs = max(1, total // 50)  # 50 chunks for progress updates
    job_chunks = [all_jobs[i:i+cs] for i in range(0, total, cs)]
    progress_bar.progress(0.02); t0 = time.time()
    all_results = []
    for ch in job_chunks:
        all_results.extend(worker_chunk(ch))
        done = len(all_results); pct = done/total; el = time.time()-t0
        eta = (el/pct*(1-pct)) if pct > 0.05 else 0
        progress_bar.progress(min(pct,0.99))
        status_text.text(f"⏳ {done:,.0f}/{total:,.0f} ({pct:.1%}) — {el:.0f}s — ETA {eta:.0f}s")
    progress_bar.progress(1.0); el = time.time()-t0
    status_text.text(f"✅ {len(all_results):,.0f} sims — {el:.0f}s ({el/60:.1f}min) — {len(all_results)/max(el,0.1):,.0f}/s")
    df = pd.DataFrame(np.array(all_results, dtype=np.float32), columns=PHYS_COLS)
    for c in ['sim_weeks','planning_freq_wks','smart_allocation','lt_raw_material','lt_semifinished',
              'lt_finished_product','lt_distribution','lt_total_weeks','pct_in_stores','pct_in_warehouse',
              'pct_in_semifinished','pct_in_raw_material','weeks_with_stockout','initial_stock_total',
              'demand_split_store_a_pct']:
        df[c] = df[c].astype(int)
    return df

def expand_financials(df_phys, price, var_cost, fixed_pcts, base_forecast, max_rows=3_000_000):
    n = len(fixed_pcts)
    df_s = df_phys.sample(max_rows//n, random_state=42) if len(df_phys)*n > max_rows else df_phys
    rows = []
    for fp in fixed_pcts:
        sub = df_s.copy()
        sub['selling_price'] = price; sub['variable_cost_per_unit'] = var_cost; sub['fixed_cost_pct'] = fp
        sub['total_revenue'] = sub['units_sold'] * price
        sub['total_variable_cost'] = (sub['units_produced'] + sub['initial_stock_total']) * var_cost
        sub['initial_stock_cost'] = sub['initial_stock_total'] * var_cost
        sub['gross_profit'] = sub['total_revenue'] - sub['total_variable_cost']
        sub['total_fixed_cost'] = base_forecast * 52 * price * fp * (sub['sim_weeks'] / 52)
        sub['net_profit'] = sub['gross_profit'] - sub['total_fixed_cost']
        sub['net_margin_pct'] = sub['net_profit'] / sub['total_revenue'].replace(0, np.nan)
        sub['service_level'] = sub['units_sold'] / sub['units_demanded'].replace(0, np.nan)
        sub['lost_sales_revenue'] = sub['units_missed'] * price
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)

# ════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════
def label_strategies(df):
    mlt = df['lt_total_weeks'].median()
    df['stock_strategy'] = np.where(df['pct_in_stores']>=80,'🏪 Centralized (≥80% in stores)','🔀 Distributed (<80% in stores)')
    df['speed_strategy'] = np.where(df['lt_total_weeks']<mlt,'⚡ Fast SC (LT<'+str(int(mlt))+'wk)','🐢 Slow SC (LT≥'+str(int(mlt))+'wk)')
    df['strategy_quadrant'] = df['stock_strategy']+' × '+df['speed_strategy']
    return df, mlt

def run_analysis(df_full):
    dc = df_full.dropna(subset=['net_margin_pct']).copy()
    dc['smart_allocation'] = dc['smart_allocation'].astype(int)
    X = dc[ML_FEATURES].values; y = dc['net_margin_pct'].values
    w = dc['demand_weight'].values if 'demand_weight' in dc.columns else None
    if len(X)>2_000_000:
        idx = np.random.RandomState(42).choice(len(X),2_000_000,replace=False)
        X,y = X[idx],y[idx]
        if w is not None: w = w[idx]
    pretty_names = [PRETTY.get(f,f) for f in ML_FEATURES]
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=500, random_state=42)
    tree.fit(X,y,sample_weight=w)
    rf = RandomForestRegressor(n_estimators=100,max_depth=8,min_samples_leaf=200,random_state=42,n_jobs=1)
    rf.fit(X,y,sample_weight=w)
    corr = {f: np.corrcoef(X[:,i],y)[0,1] for i,f in enumerate(ML_FEATURES)}
    fig,ax = plt.subplots(figsize=(28,12),dpi=100)
    plot_tree(tree,feature_names=pretty_names,filled=True,rounded=True,ax=ax,fontsize=8,impurity=False,proportion=True,label='root',precision=2)
    ax.set_title("Regression Tree — Net Margin % Drivers",fontsize=14,fontweight='bold'); plt.tight_layout()
    return {'tree_r2':tree.score(X,y),'tree_text':export_text(tree,feature_names=pretty_names,max_depth=4),
            'tree_imp':dict(zip(ML_FEATURES,tree.feature_importances_)),'tree_fig':fig,
            'rf_r2':rf.score(X,y),'rf_imp':dict(zip(ML_FEATURES,rf.feature_importances_)),
            'features':ML_FEATURES,'n_samples':len(X),'correlations':corr}

def strategy_matrix(df):
    has_w = 'demand_weight' in df.columns
    results = {}
    for quad in df['strategy_quadrant'].unique():
        g = df[df['strategy_quadrant']==quad]
        if len(g)==0: continue
        results[quad] = {
            'Scenarios':len(g),
            'Weighted Avg Net Margin %':weighted_mean(g,'net_margin_pct') if has_w else g['net_margin_pct'].mean(),
            'Median Net Margin %':g['net_margin_pct'].median(),
            'Weighted Avg Service Level':weighted_mean(g,'service_level') if has_w else g['service_level'].mean(),
            '% Scenarios Profitable':(g['net_margin_pct']>0).mean(),
            'Weighted Avg Net Profit €':weighted_mean(g,'net_profit') if has_w else g['net_profit'].mean(),
            'Std Margin %':g['net_margin_pct'].std(),
            'Avg Initial Stock':g['initial_stock_total'].mean(),
            'Avg Total LT (wks)':g['lt_total_weeks'].mean(),
            'Avg % in Stores':g['pct_in_stores'].mean(),
        }
    return pd.DataFrame(results).T

def paired_comparison(df):
    has_w = 'demand_weight' in df.columns
    mlt = df['lt_total_weeks'].median()
    cs = df[(df['pct_in_stores']>=80)&(df['lt_total_weeks']>=mlt)]
    dfa = df[(df['pct_in_stores']<80)&(df['lt_total_weeks']<mlt)]
    mk = ['demand_level','sim_weeks','initial_stock_total','fixed_cost_pct']
    cs_a = cs.groupby(mk)['net_margin_pct'].mean().reset_index(); cs_a.columns = mk+['margin_cent']
    df_a = dfa.groupby(mk)['net_margin_pct'].mean().reset_index(); df_a.columns = mk+['margin_dist']
    p = cs_a.merge(df_a,on=mk,how='inner')
    if len(p)==0: return {'pairs':0,'pct_dist':0,'pct_cent':0,'delta':0}
    if has_w:
        wmap = dict(zip(df['demand_level'].unique(),[df[df['demand_level']==dm]['demand_weight'].iloc[0] for dm in df['demand_level'].unique()]))
        p['w'] = p['demand_level'].map(wmap).fillna(1)
    else: p['w'] = 1.0
    ws = p['w'].sum()
    dw = ((p['margin_dist']>p['margin_cent']).astype(float)*p['w']).sum()/ws
    delta = ((p['margin_dist']-p['margin_cent'])*p['w']).sum()/ws
    return {'pairs':len(p),'pct_dist':dw,'pct_cent':1-dw,'delta':delta}

def smart_comparison(df):
    ag = df[df['pct_in_stores']<80]; has_w = 'demand_weight' in ag.columns
    on = ag[ag['smart_allocation']==1]; off = ag[ag['smart_allocation']==0]
    return {
        'sm_m':weighted_mean(on,'net_margin_pct') if has_w and len(on)>0 else (on['net_margin_pct'].mean() if len(on)>0 else 0),
        'pu_m':weighted_mean(off,'net_margin_pct') if has_w and len(off)>0 else (off['net_margin_pct'].mean() if len(off)>0 else 0),
        'sm_s':weighted_mean(on,'service_level') if has_w and len(on)>0 else (on['service_level'].mean() if len(on)>0 else 0),
        'pu_s':weighted_mean(off,'service_level') if has_w and len(off)>0 else (off['service_level'].mean() if len(off)>0 else 0),
    }

# ════════════════════════════════════════════════════════════════
# HTML REPORT GENERATOR
# ════════════════════════════════════════════════════════════════
def generate_html_report(df_full, analysis, matrix_df, pairs, smart, median_lt,
                         params, weights, demand_mults, center, sigma, price, var_cost,
                         fixed_pcts, stock_levels, demand_splits, sim_weeks, stock_distribs, lt_combos, ex):
    buf = io.BytesIO()
    analysis['tree_fig'].savefig(buf,format='png',dpi=150,bbox_inches='tight')
    tree_b64 = base64.b64encode(buf.getvalue()).decode(); buf.close()
    fig_dem,ax_dem = plt.subplots(figsize=(8,2.5),dpi=100)
    ax_dem.bar(demand_mults,[w*100 for w in weights],width=0.06,color='#4a90d9')
    ax_dem.set_xlabel('Demand ×'); ax_dem.set_ylabel('Probability %')
    ax_dem.set_title(f'Demand Profile (center={center:.0%}, σ={sigma})'); plt.tight_layout()
    buf2 = io.BytesIO()
    fig_dem.savefig(buf2,format='png',dpi=100,bbox_inches='tight')
    dem_b64 = base64.b64encode(buf2.getvalue()).decode(); buf2.close(); plt.close(fig_dem)

    def df_to_html(df, pct_cols=None, eur_cols=None, int_cols=None):
        pct_cols=pct_cols or []; eur_cols=eur_cols or []; int_cols=int_cols or []
        h='<table><thead><tr>'
        for c in df.columns: h+=f'<th>{c}</th>'
        h+='</tr></thead><tbody>'
        for _,row in df.iterrows():
            h+='<tr>'
            for c in df.columns:
                v=row[c]
                if c in pct_cols and isinstance(v,(int,float,np.floating)): h+=f'<td>{v:.1%}</td>'
                elif c in eur_cols and isinstance(v,(int,float,np.floating)): h+=f'<td>€{v:,.0f}</td>'
                elif c in int_cols and isinstance(v,(int,float,np.floating)): h+=f'<td>{v:,.0f}</td>'
                else: h+=f'<td>{v}</td>'
            h+='</tr>'
        h+='</tbody></table>'; return h

    def build_breakdown(df_src,group_col,label):
        rows=[]
        for val in sorted(df_src[group_col].unique()):
            g=df_src[df_src[group_col]==val]
            rows.append({label:int(val) if isinstance(val,(int,float,np.integer,np.floating)) else val,
                'Wtd Avg Net Margin %':weighted_mean(g,'net_margin_pct'),
                'Wtd Avg Service Level':weighted_mean(g,'service_level'),
                'Wtd Avg Net Profit €':weighted_mean(g,'net_profit')})
        return pd.DataFrame(rows)

    bd_stock=build_breakdown(df_full,'initial_stock_total','Initial Stock')
    bd_store=build_breakdown(df_full,'pct_in_stores','% in Stores')
    bd_lt=build_breakdown(df_full,'lt_total_weeks','Total LT (wks)')
    bd_wh=build_breakdown(df_full,'pct_in_warehouse','% in Warehouse') if df_full['pct_in_warehouse'].nunique()>1 else None
    bd_semi=build_breakdown(df_full,'pct_in_semifinished','% in Semi-Finished') if df_full['pct_in_semifinished'].nunique()>1 else None
    bd_split=build_breakdown(df_full,'demand_split_store_a_pct','Store A Demand %') if len(demand_splits)>1 else None

    imp_html_data=[]
    for f in ML_FEATURES:
        rf_imp=analysis['rf_imp'].get(f,0)
        if rf_imp<0.005: continue
        corr_v=analysis['correlations'][f]
        imp_html_data.append({'Feature':PRETTY.get(f,f),'Forest Importance':rf_imp,
            'Tree Importance':analysis['tree_imp'].get(f,0),
            'Direction':'📈 Increases margin' if corr_v>0.01 else '📉 Decreases margin' if corr_v<-0.01 else '↔️ Neutral',
            'Correlation':corr_v})
    imp_html_df=pd.DataFrame(imp_html_data).sort_values('Forest Importance',ascending=False)
    sm_df=matrix_df.copy().reset_index().rename(columns={'index':'Strategy'})
    wm=weighted_mean(df_full,'net_margin_pct'); ws=weighted_mean(df_full,'service_level'); ppos=(df_full['net_margin_pct']>0).mean()
    dw=pairs['pct_dist']*100; cw=pairs['pct_cent']*100; sm_d=smart['sm_m']-smart['pu_m']
    t5=sorted(analysis['rf_imp'].items(),key=lambda x:-x[1])[:5]
    pos_f=[(f,analysis['correlations'][f]) for f in ML_FEATURES if analysis['rf_imp'][f]>0.01 and analysis['correlations'][f]>0.01]
    neg_f=[(f,analysis['correlations'][f]) for f in ML_FEATURES if analysis['rf_imp'][f]>0.01 and analysis['correlations'][f]<-0.01]
    pos_f.sort(key=lambda x:-analysis['rf_imp'][x[0]]); neg_f.sort(key=lambda x:-analysis['rf_imp'][x[0]])
    verdict_class='verdict' if dw>55 else 'verdict mixed' if dw>45 else 'verdict push'

    html=f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>SC Agility — Results</title>
<style>
body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px 40px;background:#f4f6f9;color:#1a2a40}}
h1{{color:#1a2a40;border-bottom:3px solid #4a90d9;padding-bottom:10px}}
h2{{color:#2a3a50;margin-top:35px;border-bottom:2px solid #e0e4ea;padding-bottom:6px}}h3{{color:#3a4a60;margin-top:20px}}
.metrics{{display:flex;gap:15px;flex-wrap:wrap;margin:15px 0}}
.metric{{background:linear-gradient(135deg,#fff,#f7f9fc);border:1px solid #dde3ed;border-radius:10px;padding:15px 20px;text-align:center;flex:1;min-width:150px;box-shadow:0 1px 4px rgba(0,0,0,0.05)}}
.metric .value{{font-size:26px;font-weight:800;color:#1a2a40}}.metric .label{{font-size:11px;color:#7a8a9e;text-transform:uppercase;letter-spacing:1px;margin-top:4px}}
.metric.green .value{{color:#28a745}}.metric.red .value{{color:#dc3545}}.metric.blue .value{{color:#4a90d9}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
th{{background:#2a3a50;color:white;padding:10px 12px;text-align:left;font-size:12px}}
td{{padding:8px 12px;border-bottom:1px solid #e0e4ea}}tr:nth-child(even){{background:#f8f9fb}}tr:hover{{background:#eef1f5}}
.box{{background:white;border:1px solid #dce2ea;border-radius:10px;padding:20px;margin:15px 0}}
.verdict{{background:linear-gradient(135deg,#e8f5e9,#f1f8e9);border-left:5px solid #28a745;padding:15px 20px;border-radius:8px;margin:15px 0;font-size:15px}}
.verdict.mixed{{background:linear-gradient(135deg,#fff8e1,#fff3e0);border-left-color:#ff9800}}
.verdict.push{{background:linear-gradient(135deg,#fce4ec,#fff3e0);border-left-color:#dc3545}}
img{{max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin:10px 0}}
.cols{{display:flex;gap:20px;flex-wrap:wrap}}.col{{flex:1;min-width:300px}}
.timestamp{{color:#999;font-size:12px}}.param{{display:inline-block;background:#eef1f5;padding:3px 10px;border-radius:12px;margin:2px;font-size:12px}}
</style></head><body>
<h1>📊 SC Agility — Batch Simulation Results</h1>
<p class="timestamp">Generated: {time.strftime('%Y-%m-%d %H:%M')} · {len(df_full):,.0f} scenarios</p>
<h2>⚙️ Parameters</h2><div class="box">
<span class="param">💰 Price: €{price:,}</span><span class="param">🏭 Var cost: €{var_cost:,}/unit</span>
<span class="param">📦 Stock: {stock_levels}</span><span class="param">📈 Demand center: {center:.0%}</span>
<span class="param">📊 σ = {sigma}</span><span class="param">🏪 Splits A: {demand_splits}%</span>
<span class="param">📅 Weeks: {sim_weeks}</span><span class="param">📦 {len(stock_distribs)} distributions</span>
<span class="param">⏱️ {len(lt_combos)} LT combos</span><span class="param">💰 Fixed costs: {[f'{f:.0%}' for f in fixed_pcts]}</span>
</div>
<h2>📈 Demand Profile</h2><img src="data:image/png;base64,{dem_b64}" alt="Demand Profile">
<h2>📊 Key Metrics (demand-weighted)</h2><div class="metrics">
<div class="metric"><div class="value">{len(df_full):,.0f}</div><div class="label">Total Scenarios</div></div>
<div class="metric blue"><div class="value">{wm:.1%}</div><div class="label">Wtd Avg Net Margin</div></div>
<div class="metric"><div class="value">{ws:.1%}</div><div class="label">Wtd Avg Service Level</div></div>
<div class="metric {'green' if ppos>0.7 else 'red'}"><div class="value">{ppos:.1%}</div><div class="label">% Profitable</div></div>
</div>
<h2>🌳 Regression Tree — Net Margin Drivers</h2><img src="data:image/png;base64,{tree_b64}" alt="Tree">
<h2>🔑 Feature Importance + Direction</h2>{df_to_html(imp_html_df,pct_cols=['Forest Importance','Tree Importance'])}
<h2>⚔️ Strategy Matrix: Stock Position × SC Speed</h2>
<p><em>Centralized = ≥80% in stores · Fast = LT &lt; {int(median_lt)} wks</em></p>
{df_to_html(sm_df,pct_cols=['Weighted Avg Net Margin %','Median Net Margin %','Weighted Avg Service Level','% Scenarios Profitable','Std Margin %'],eur_cols=['Weighted Avg Net Profit €'],int_cols=['Scenarios','Avg Initial Stock'])}
<h2>🏆 Head-to-Head</h2><div class="metrics">
<div class="metric"><div class="value">{pairs['pairs']:,.0f}</div><div class="label">Matched Pairs</div></div>
<div class="metric green"><div class="value">{pairs['pct_dist']:.1%}</div><div class="label">Distributed+Fast Wins</div></div>
<div class="metric red"><div class="value">{pairs['pct_cent']:.1%}</div><div class="label">Centralized+Slow Wins</div></div>
<div class="metric blue"><div class="value">{pairs['delta']:+.1%}</div><div class="label">Wtd Δ Margin</div></div>
</div>
<h2>🧠 Smart Allocation vs Push 50/50</h2><div class="metrics">
<div class="metric"><div class="value">{smart['sm_m']:.1%}</div><div class="label">Smart Avg Margin</div></div>
<div class="metric"><div class="value">{smart['pu_m']:.1%}</div><div class="label">Push Avg Margin</div></div>
<div class="metric"><div class="value">{smart['sm_s']:.1%}</div><div class="label">Smart Avg Service</div></div>
<div class="metric"><div class="value">{smart['pu_s']:.1%}</div><div class="label">Push Avg Service</div></div>
</div>
<h2>📊 Breakdowns</h2><div class="cols"><div class="col"><h3>By Stock Level</h3>
{df_to_html(bd_stock,pct_cols=['Wtd Avg Net Margin %','Wtd Avg Service Level'],eur_cols=['Wtd Avg Net Profit €'])}
</div><div class="col"><h3>By % in Stores</h3>
{df_to_html(bd_store,pct_cols=['Wtd Avg Net Margin %','Wtd Avg Service Level'],eur_cols=['Wtd Avg Net Profit €'])}
</div></div><div class="cols"><div class="col"><h3>By Total Lead Time</h3>
{df_to_html(bd_lt,pct_cols=['Wtd Avg Net Margin %','Wtd Avg Service Level'],eur_cols=['Wtd Avg Net Profit €'])}
</div><div class="col">"""

    if bd_wh is not None:
        html+=f"""<h3>By % in Warehouse</h3>{df_to_html(bd_wh,pct_cols=['Wtd Avg Net Margin %','Wtd Avg Service Level'],eur_cols=['Wtd Avg Net Profit €'])}"""
    if bd_semi is not None:
        html+=f"""<h3>By % in Semi-Finished</h3>{df_to_html(bd_semi,pct_cols=['Wtd Avg Net Margin %','Wtd Avg Service Level'],eur_cols=['Wtd Avg Net Profit €'])}"""
    if bd_split is not None:
        html+=f"""<h3>By Demand Split (Store A %)</h3>{df_to_html(bd_split,pct_cols=['Wtd Avg Net Margin %','Wtd Avg Service Level'])}"""

    html+=f"""</div></div>
<h2>📋 Executive Summary</h2><div class="box">
<h3>Key Drivers (Random Forest R²={analysis['rf_r2']:.3f})</h3>
<table><thead><tr><th>Feature</th><th>Importance</th><th>Effect</th></tr></thead><tbody>"""
    for f,imp in t5:
        d='📈 Increases' if analysis['correlations'][f]>0.01 else '📉 Decreases' if analysis['correlations'][f]<-0.01 else '↔️'
        html+=f"<tr><td>{PRETTY.get(f,f)}</td><td>{imp:.3f}</td><td>{d} margin</td></tr>"
    html+=f"""</tbody></table>
<p><b>Positive:</b> {', '.join(PRETTY.get(f,f) for f,_ in pos_f[:4]) or 'None'}</p>
<p><b>Negative:</b> {', '.join(PRETTY.get(f,f) for f,_ in neg_f[:4]) or 'None'}</p>
<h3>Strategy — {pairs['pairs']:,.0f} matched pairs</h3>
<p><b>Distributed+Fast wins {dw:.1f}%</b> · Δ margin: <b>{pairs['delta']:+.1%}</b></p>
<p>Smart allocation: Δ margin <b>{sm_d:+.1%}</b></p>
<div class="{verdict_class}"><b>Verdict:</b>
{"Agility wins. Distributed stock + short LTs outperforms centralized push." if dw>55 else "Mixed. Neither strategy dominates." if 45<dw<55 else "Centralized holds. Pre-loading stores is competitive."}
</div></div>
<p class="timestamp" style="margin-top:40px;text-align:center">SC Agility Simulator · {time.strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html

# ════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ════════════════════════════════════════════════════════════════
st.markdown("""<style>.stApp{background:#f4f6f9}h1{color:#1a2a40}h2{color:#2a3a50;border-bottom:2px solid #e0e4ea;padding-bottom:6px}</style>""",unsafe_allow_html=True)
st.title("📊 Supply Chain Agility Simulator")
st.markdown("*Combinatorial batch analysis · Weighted demand · Push vs Agile · Visual ML analysis*")

with st.sidebar:
    st.header("🎛️ Parameters")

    st.subheader("💰 Pricing")
    price = st.number_input("Selling price €",value=1000,step=100)
    var_cost = st.number_input("Variable cost €/unit",value=200,step=50)

    st.subheader("📈 Demand Profile")
    dm_min = st.number_input("Min demand ×",value=0.30,step=0.05,format="%.2f")
    dm_max = st.number_input("Max demand ×",value=4.00,step=0.10,format="%.2f")
    dm_step = st.number_input("Step",value=0.30,step=0.05,format="%.2f",key="dms")
    demand_mults = [round(dm_min+i*dm_step,2) for i in range(int((dm_max-dm_min)/dm_step)+1)]
    demand_mults = [d for d in demand_mults if d<=dm_max+0.001]
    sigma = st.slider("σ (uncertainty spread)",0.1,1.5,0.6,0.05)
    center = st.slider("📍 Curve center (most likely demand)",0.30,1.50,0.40,0.05,
        help="0.40 = conservative forecast. 0.70 = moderate. 1.00 = optimistic.")
    default_weights = compute_lognormal_weights(demand_mults,sigma,center)
    expected_dem = sum(d*w for d,w in zip(demand_mults,default_weights))
    st.caption(f"Peak ≈ **{center:.0%}** · Expected: **{expected_dem:.2f}×**")
    with st.expander("🎚️ Edit weights"):
        ew={}
        for dm,dw in zip(demand_mults,default_weights):
            ew[dm]=st.number_input(f"{dm:.2f}×",value=round(dw*100,1),step=0.5,format="%.1f",key=f"w_{dm}_s{sigma:.2f}_c{center:.2f}")
        raw=[ew[dm] for dm in demand_mults]; tot=sum(raw)
        weights=[w/tot for w in raw] if tot>0 else default_weights
    w_df=pd.DataFrame({'Demand ×':demand_mults,'Prob %':[w*100 for w in weights]})
    st.bar_chart(w_df,x='Demand ×',y='Prob %',height=130,color='#4a90d9')

    st.subheader("🏪 Demand Split A/B")
    demand_splits = st.multiselect("Store A demand %",[50,60,70,80],default=[60,80])
    if not demand_splits: demand_splits=[50]

    st.subheader("📅 Simulation length")
    sim_weeks=[]
    if st.checkbox("13 wks",value=True): sim_weeks.append(13)
    if st.checkbox("26 wks",value=True): sim_weeks.append(26)
    if st.checkbox("39 wks",value=False): sim_weeks.append(39)
    if st.checkbox("52 wks",value=False): sim_weeks.append(52)
    if not sim_weeks: sim_weeks=[26]

    st.subheader("⏱️ Lead Times per Stage")
    lt_configs={}
    for stage,abbr,dmin,dmax,dstp in [("Raw Material","rm",1,9,4),("Semi-Finished","sf",1,9,4),("Finished Product","fp",1,9,4),("Distribution","di",1,5,2)]:
        with st.expander(f"🔧 {stage}"):
            c1,c2,c3=st.columns(3)
            lmin=c1.number_input("Min",value=dmin,min_value=1,step=1,key=f"lt_{abbr}_min")
            lmax=c2.number_input("Max",value=dmax,step=1,key=f"lt_{abbr}_max")
            lstp=c3.number_input("Step",value=dstp,min_value=1,step=1,key=f"lt_{abbr}_stp")
            lt_configs[stage]=list(range(lmin,lmax+1,lstp))
            st.caption(f"→ {lt_configs[stage]}")
    lt_combos=list(itertools.product(lt_configs["Raw Material"],lt_configs["Semi-Finished"],lt_configs["Finished Product"],lt_configs["Distribution"]))
    if len(lt_combos)>50:
        rng=np.random.RandomState(42); idx=rng.choice(len(lt_combos),50,replace=False)
        lt_combos=[lt_combos[i] for i in sorted(idx)]
        st.warning(f"Sampled 50 LT combos for performance")
    st.caption(f"**{len(lt_combos)}** LT combinations")

    st.subheader("📦 Initial Stock (absolute)")
    stk_min=st.number_input("Min stock",value=500,step=100)
    stk_max=st.number_input("Max stock",value=1300,step=100)
    stk_stp=st.number_input("Step",value=400,step=100,key="stk_s")
    stock_levels=list(range(stk_min,stk_max+1,stk_stp))
    st.caption(f"Levels: {stock_levels}")

    st.subheader("📦 Stock Distribution along SC")
    sd_c1,sd_c2,sd_c3=st.columns(3)
    with sd_c1:
        st.markdown("**🏪 Stores**")
        sd_st_min=st.number_input("Min %",value=30,step=10,key="sd_st_min")
        sd_st_max=st.number_input("Max %",value=100,step=10,key="sd_st_max")
        sd_st_stp=st.number_input("Step %",value=20,step=5,key="sd_st_stp")
    with sd_c2:
        st.markdown("**🏭 Warehouse**")
        sd_wh_min=st.number_input("Min %",value=0,step=10,key="sd_wh_min")
        sd_wh_max=st.number_input("Max %",value=30,step=10,key="sd_wh_max")
        sd_wh_stp=st.number_input("Step %",value=15,step=5,key="sd_wh_stp")
    with sd_c3:
        st.markdown("**🔧 Semi**")
        sd_se_min=st.number_input("Min %",value=0,step=10,key="sd_se_min")
        sd_se_max=st.number_input("Max %",value=30,step=10,key="sd_se_max")
        sd_se_stp=st.number_input("Step %",value=15,step=5,key="sd_se_stp")
    store_vals=list(range(sd_st_min,sd_st_max+1,max(1,sd_st_stp)))
    wh_vals=list(range(sd_wh_min,sd_wh_max+1,max(1,sd_wh_stp)))
    semi_vals=list(range(sd_se_min,sd_se_max+1,max(1,sd_se_stp)))
    stock_distribs=generate_stock_distribs(store_vals,wh_vals,semi_vals)
    with st.expander(f"📋 {len(stock_distribs)} combos"):
        st.dataframe(pd.DataFrame(stock_distribs,columns=['Stores%','WH%','Semi%','RM%']),hide_index=True,height=200)
    if not stock_distribs: stock_distribs=[(100,0,0,0)]

    st.subheader("🔄 Planning")
    pf=[]
    if st.checkbox("Every week",value=True): pf.append(1)
    if st.checkbox("Every 4 weeks",value=True): pf.append(4)
    if not pf: pf=[1]

    st.subheader("🏭 Capacity")
    cr_min=st.number_input("Min ramp %/wk",value=5,step=5)
    cr_max=st.number_input("Max ramp %/wk",value=25,step=5)
    cr_stp=st.number_input("Ramp step %",value=10,step=5,key="cr_s")
    cap_ramps=[r/100 for r in range(cr_min,cr_max+1,cr_stp)]
    cap_start=st.number_input("Initial capacity/wk",value=100,step=25)
    base_forecast=st.number_input("Base forecast/wk",value=100,step=25)

    st.subheader("📊 Fixed costs")
    fx_min=st.number_input("Min fixed %",value=20,step=5)
    fx_max=st.number_input("Max fixed %",value=40,step=5)
    fx_stp=st.number_input("Fixed step %",value=10,step=5,key="fx_s")
    fixed_pcts=[f/100 for f in range(fx_min,fx_max+1,fx_stp)]

    st.subheader("🧠 Distribution")
    smart_opts=[]
    if st.checkbox("Push 50/50",value=True): smart_opts.append(False)
    if st.checkbox("Smart allocation",value=True): smart_opts.append(True)
    if not smart_opts: smart_opts=[False]

# ─── SCENARIO COUNT ───
params={'demand_mults':demand_mults,'sim_weeks':sim_weeks,'plan_freqs':pf,'cap_ramps':cap_ramps,
    'smart_opts':smart_opts,'lt_combos':lt_combos,'stock_levels':stock_levels,
    'stock_distribs':stock_distribs,'cap_start':cap_start,'base_forecast':base_forecast,'demand_splits':demand_splits}
n_phys=(len(demand_mults)*len(sim_weeks)*len(pf)*len(cap_ramps)*len(smart_opts)*len(lt_combos)*len(stock_levels)*len(stock_distribs)*len(demand_splits))
n_fin=n_phys*len(fixed_pcts)
est=n_phys*65e-6/60

if n_phys > 5_000_000:
    st.warning(f"⚠️ {n_phys:,.0f} sims may take >10 min on Streamlit Cloud. Increase step sizes to reduce.")

st.info(f"""
**{n_phys:,.0f}** physical sims × {len(fixed_pcts)} costs = **{n_fin:,.0f}** scenarios  
📈 Center **{center:.0%}**, σ={sigma}, expected={sum(d*w for d,w in zip(demand_mults,weights)):.2f}×  
⏱️ Estimated ~**{max(1,est):.0f} min**
""")

# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════
if st.button("🚀 Run All Simulations",type="primary",use_container_width=True):
    st.divider()
    st.subheader("Phase 1: Physical Simulations")
    pb=st.progress(0); stxt=st.empty(); t0=time.time()
    df_phys=run_batch(params,pb,stxt)

    st.subheader("Phase 2: Financial Expansion")
    with st.spinner("..."):
        df_full=expand_financials(df_phys,price,var_cost,fixed_pcts,base_forecast)
        df_full['demand_weight']=df_full['demand_level'].map(dict(zip(demand_mults,weights))).fillna(0)

    st.subheader("Phase 3: Strategy Labeling")
    with st.spinner("..."): df_full,median_lt=label_strategies(df_full)

    st.subheader("Phase 4: ML Analysis")
    with st.spinner("Training..."): analysis=run_analysis(df_full)
    st.success(f"✅ Tree R²={analysis['tree_r2']:.3f}, Forest R²={analysis['rf_r2']:.3f}")

    st.subheader("Phase 5: Strategy Comparison")
    with st.spinner("..."):
        matrix_df=strategy_matrix(df_full); pairs=paired_comparison(df_full); smart=smart_comparison(df_full)

    tt=time.time()-t0; st.success(f"🏁 **{tt:.0f}s ({tt/60:.1f} min)**")

    # ─── RESULTS ───
    st.divider(); st.header("📈 Results (demand-weighted)")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Scenarios",f"{len(df_full):,.0f}")
    c2.metric("Wtd Avg Net Margin",f"{weighted_mean(df_full,'net_margin_pct'):.1%}")
    c3.metric("Wtd Avg Service",f"{weighted_mean(df_full,'service_level'):.1%}")
    c4.metric("% Profitable",f"{(df_full['net_margin_pct']>0).mean():.1%}")

    st.subheader("🌳 Decision Tree — What Drives Net Margin?")
    st.pyplot(analysis['tree_fig'],use_container_width=True); plt.close(analysis['tree_fig'])
    with st.expander("📝 Tree Rules"): st.code(analysis['tree_text'],language='text')

    st.subheader("🔑 Feature Importance")
    imp_data=[]
    for f in ML_FEATURES:
        imp_data.append({'Feature':PRETTY.get(f,f),'Tree':analysis['tree_imp'].get(f,0),
            'Forest':analysis['rf_imp'].get(f,0),
            'Effect':'📈 +margin' if analysis['correlations'][f]>0.01 else '📉 −margin' if analysis['correlations'][f]<-0.01 else '↔️',
            'Correlation':analysis['correlations'][f]})
    imp_df=pd.DataFrame(imp_data).sort_values('Forest',ascending=False).query('Forest>0.005')
    st.dataframe(imp_df.style.format({'Tree':'{:.3f}','Forest':'{:.3f}','Correlation':'{:+.3f}'}).bar(subset=['Forest'],color='#4a90d9'),hide_index=True,use_container_width=True)

    st.subheader("⚔️ Strategy Matrix")
    st.caption(f"Centralized = ≥80% in stores · Fast = LT < {int(median_lt)} wks")
    fmt_m=matrix_df.copy()
    for col in ['Weighted Avg Net Margin %','Median Net Margin %','Weighted Avg Service Level','% Scenarios Profitable','Std Margin %']:
        if col in fmt_m.columns: fmt_m[col]=fmt_m[col].map(lambda x:f"{x:.1%}")
    for col in ['Weighted Avg Net Profit €']:
        if col in fmt_m.columns: fmt_m[col]=fmt_m[col].map(lambda x:f"€{x:,.0f}")
    for col in ['Scenarios','Avg Initial Stock']:
        if col in fmt_m.columns: fmt_m[col]=fmt_m[col].map(lambda x:f"{x:,.0f}")
    if 'Avg Total LT (wks)' in fmt_m.columns: fmt_m['Avg Total LT (wks)']=fmt_m['Avg Total LT (wks)'].map(lambda x:f"{x:.0f}")
    if 'Avg % in Stores' in fmt_m.columns: fmt_m['Avg % in Stores']=fmt_m['Avg % in Stores'].map(lambda x:f"{x:.0f}%")
    st.dataframe(fmt_m,use_container_width=True)

    st.subheader("🏆 Head-to-Head")
    h1,h2,h3,h4=st.columns(4)
    h1.metric("Matched Pairs",f"{pairs['pairs']:,.0f}")
    h2.metric("Distributed+Fast",f"{pairs['pct_dist']:.1%}")
    h3.metric("Centralized+Slow",f"{pairs['pct_cent']:.1%}")
    h4.metric("Wtd Δ Margin",f"{pairs['delta']:+.1%}")

    st.subheader("🧠 Smart vs Push")
    s1,s2,s3,s4=st.columns(4)
    s1.metric("Smart Margin",f"{smart['sm_m']:.1%}"); s2.metric("Push Margin",f"{smart['pu_m']:.1%}")
    s3.metric("Smart Svc",f"{smart['sm_s']:.1%}"); s4.metric("Push Svc",f"{smart['pu_s']:.1%}")

    # Breakdowns
    def show_breakdown(df_src,group_col,title):
        rows=[]
        for val in sorted(df_src[group_col].unique()):
            g=df_src[df_src[group_col]==val]
            rows.append({title:int(val),'Wtd Margin':weighted_mean(g,'net_margin_pct'),
                'Wtd Svc':weighted_mean(g,'service_level'),'Wtd Profit €':weighted_mean(g,'net_profit')})
        st.dataframe(pd.DataFrame(rows).set_index(title).style.format({'Wtd Margin':'{:.1%}','Wtd Svc':'{:.1%}','Wtd Profit €':'€{:,.0f}'}),use_container_width=True)

    st.subheader("📊 By Stock Level"); show_breakdown(df_full,'initial_stock_total','Stock')
    st.subheader("📊 By % in Stores"); show_breakdown(df_full,'pct_in_stores','% Stores')
    if df_full['pct_in_warehouse'].nunique()>1:
        st.subheader("📊 By % in Warehouse"); show_breakdown(df_full,'pct_in_warehouse','% WH')
    if df_full['pct_in_semifinished'].nunique()>1:
        st.subheader("📊 By % in Semi-Finished"); show_breakdown(df_full,'pct_in_semifinished','% Semi')
    st.subheader("📊 By Total Lead Time"); show_breakdown(df_full,'lt_total_weeks','LT (wks)')
    if len(demand_splits)>1:
        st.subheader("📊 By Demand Split"); show_breakdown(df_full,'demand_split_store_a_pct','Store A %')

    # Executive summary
    st.divider(); st.header("📋 Executive Summary")
    t5=sorted(analysis['rf_imp'].items(),key=lambda x:-x[1])[:5]
    dw=pairs['pct_dist']*100; cw=pairs['pct_cent']*100; sm_d=smart['sm_m']-smart['pu_m']
    pos_f=[(f,analysis['correlations'][f]) for f in ML_FEATURES if analysis['rf_imp'][f]>0.01 and analysis['correlations'][f]>0.01]
    neg_f=[(f,analysis['correlations'][f]) for f in ML_FEATURES if analysis['rf_imp'][f]>0.01 and analysis['correlations'][f]<-0.01]
    pos_f.sort(key=lambda x:-analysis['rf_imp'][x[0]]); neg_f.sort(key=lambda x:-analysis['rf_imp'][x[0]])
    ex=f"""
### Setup
**{len(df_full):,.0f}** scenarios · €{price} price · €{var_cost} var cost · Stock {stock_levels}
· Demand center={center:.0%}, σ={sigma} · Initial stock cost included

### Key Drivers (RF R²={analysis['rf_r2']:.3f})
| Feature | Importance | Effect |
|---------|-----------|--------|
"""
    for f,imp in t5:
        d='📈 Increases' if analysis['correlations'][f]>0.01 else '📉 Decreases' if analysis['correlations'][f]<-0.01 else '↔️'
        ex+=f"| {PRETTY.get(f,f)} | {imp:.3f} | {d} margin |\n"
    ex+=f"""
### Positive: {', '.join(PRETTY.get(f,f) for f,_ in pos_f[:4]) or 'None'}
### Negative: {', '.join(PRETTY.get(f,f) for f,_ in neg_f[:4]) or 'None'}

### Strategy — {pairs['pairs']:,.0f} matched pairs
**Distributed+Fast wins {dw:.1f}%** · Δ margin: **{pairs['delta']:+.1%}**
Smart allocation: Δ margin **{sm_d:+.1%}**

### Verdict
{"**Agility wins.** Distributed + short LTs outperforms." if dw>55 else "**Mixed.** Depends on parameters." if 45<dw<55 else "**Centralized holds.** Pre-loading stores is competitive."}
"""
    st.markdown(ex)

    # Export HTML
    st.divider(); st.subheader("💾 Export")
    html_report=generate_html_report(df_full,analysis,matrix_df,pairs,smart,median_lt,
        params,weights,demand_mults,center,sigma,price,var_cost,fixed_pcts,stock_levels,
        demand_splits,sim_weeks,stock_distribs,lt_combos,ex)
    st.download_button("📄 Download HTML Report",html_report,"sc_results_report.html","text/html",use_container_width=True)

    csv_buf=df_full.sample(min(500_000,len(df_full)),random_state=42).to_csv(index=False) if len(df_full)>500_000 else df_full.to_csv(index=False)
    st.download_button("📊 Download Financial CSV",csv_buf,"sc_financial_results.csv","text/csv",use_container_width=True)

else:
    st.markdown("---")
    st.markdown("👈 **Configure in sidebar**, then press **Run**.")
    st.markdown("""
    **Features:**
    📈 Demand weighted by log-normal (adjustable center + σ) ·
    📦 Absolute stock levels (fair comparison) ·
    🏪 Store A/B demand split ·
    📦 Configurable stock distribution along SC ·
    🌳 Visual regression tree ·
    ⚔️ Clear 2×2 strategy matrix ·
    💾 Export to HTML report
    """)
