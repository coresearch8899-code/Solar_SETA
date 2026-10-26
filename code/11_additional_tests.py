"""Step 11 (Sections 3.1, 3.2, 3.6, 3.7; Tables 3, 6, 7): additional tests of the energy-peak divergence that need no new data.
(1) measured marginal capacity credit: PV output at the time of the monthly net peak;
(2) measured-data PV scaling at present climate (no load model): import and net-peak response to k;
(3) persistence of (1)-(2) in each complete year;
(4) month-level bootstrap of k*_imp and k*_peak at +2 degC and of their difference (reference model);
(5) alternative peak definitions (mean of the 3 largest on-peak 30-min values per month; all-hours monthly maximum);
(6) agreement of the 30-min net-peak proxy with the billed 15-min demand."""
import json, numpy as np, pandas as pd
from scipy import stats
from paths import code_src
src = code_src('09_import_multiplier.py').split('yrs = len(set(pk.month))/12')[0]
exec(src)                                   # S (42 complete months), M, FAM, TN0, GAM, Tc0, KS, kstar
from pea_invoices import INVOICES
OUT = {}; KWP = 2513.2
on = S.on.values.astype(bool); mon = S.month.astype(str).values; L0 = S.L.values; E0 = S.E.values
# (1) marginal capacity credit from measured data
pkm = pd.read_csv('peaks_monthly.csv'); H = pd.read_pickle('hosp.pkl')
EtB = H.E.reindex(pd.to_datetime(pkm.tB)).values*2         # PV power in the net-peak interval (kW)
EtL = pkm.E_at_Lpk.values
pkm['EtB'] = EtB; pkm['mcc'] = EtB/KWP*100
cf = H.E.sum()/H.E.notna().sum()*2/KWP*100                 # mean PV power / kWp (%)
OUT['marginal'] = dict(ECC=float(pkm.dP.mean()/KWP*100), PV_at_gross_peak_pct=float(np.nanmean(EtL)/KWP*100),
                       PV_at_net_peak_pct=float(np.nanmean(EtB)/KWP*100), PV_at_net_peak_median_pct=float(np.nanmedian(EtB)/KWP*100),
                       capacity_factor_pct=float(cf), share_net_peak_PV_below_10pct=float(np.mean(EtB/KWP < 0.10)*100),
                       wilcoxon_p=float(stats.wilcoxon(EtL - EtB).pvalue))
# (2) measured-data scaling at present climate (load and PV as measured; PV scaled by k; surplus curtailed)
def resp(Lv, Ev, onm, monm, ks):
    r = []
    for k in ks:
        imp = np.clip(Lv - k*Ev, 0, None)
        r.append(dict(k=k, imp=imp.sum(), peak=pd.Series(imp[onm]*2).groupby(monm[onm]).max().mean(),
                      curt=np.clip(k*Ev - Lv, 0, None).sum()/(k*Ev).sum()*100))
    return pd.DataFrame(r)
ks = [0, 1, 1.5, 2, 2.5, 3, 4]
R0 = resp(L0, E0, on, mon, ks); yrs = len(set(mon))/12
R0['imp_GWh_yr'] = R0.imp/1e6/yrs
R0['d_imp_per_MWp'] = -R0.imp_GWh_yr.diff()/(R0.k.diff()*KWP/1000)          # GWh/yr per added MWp
R0['d_peak_per_MWp'] = -R0.peak.diff()/(R0.k.diff()*KWP/1000)              # kW per added MWp
R0.to_csv('tab_measured_scaling.csv', index=False); print(R0.round(3).to_string())
OUT['measured_scaling'] = R0.to_dict('records')
# (3) by complete year
yr = S.index.year.values; OUT['by_year'] = {}
for y in [2024, 2025]:
    sel = yr == y; Ry = resp(L0[sel], E0[sel], on[sel], mon[sel], [0, 1, 2, 3])
    py = pkm[pkm.month.str[:4] == str(y)]
    OUT['by_year'][y] = dict(peak=Ry.peak.tolist(), imp=(Ry.imp/1e6).tolist(), ECC=float(py.dP.mean()/KWP*100),
                             PV_at_net_peak_pct=float(py.EtB.mean()/KWP*100), share_after15=float((py.hB >= 15).mean()*100))
print(OUT['by_year'])
# (4) month bootstrap of the multipliers (reference model)
ref = FAM['GBM, monotone (reference)'](TN0).fit(M)
def monthly(dT):
    X = S.copy(); X[TFE] = X[TFE] + dT
    L = np.clip(L0 + ref.predict(X) - ref.predict(S), 0, None); fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25))
    imp_m, pk_m = [], []
    for k in KS:
        imp = np.clip(L - E0*k*fac, 0, None)
        imp_m.append(pd.Series(imp).groupby(mon).sum()); pk_m.append(pd.Series(imp[on]*2).groupby(mon[on]).max())
    return pd.concat(imp_m, axis=1).values, pd.concat(pk_m, axis=1).values      # (months x k)
I0, P0 = monthly(0.0); I2, P2 = monthly(2.0)
def kst(base, curve):
    i = np.where(curve <= base)[0]
    if len(i) == 0: return np.nan
    i = i[0]; return float(KS[0]) if i == 0 else float(np.interp(base, [curve[i], curve[i-1]], [KS[i], KS[i-1]]))
rng = np.random.default_rng(2026); nm = I0.shape[0]; bs = []
for b in range(2000):
    s = rng.integers(0, nm, nm)
    ki = kst(I0[s, 0].sum(), I2[s].sum(axis=0)); kp = kst(P0[s, 0].mean(), P2[s].mean(axis=0)); bs.append((ki, kp))
bs = np.array(bs); ok = ~np.isnan(bs).any(axis=1)
OUT['month_bootstrap_2C'] = dict(k_imp=[float(np.percentile(bs[ok, 0], q)) for q in (2.5, 50, 97.5)],
                                 k_peak=[float(np.percentile(bs[ok, 1], q)) for q in (2.5, 50, 97.5)],
                                 diff=[float(np.percentile(bs[ok, 1] - bs[ok, 0], q)) for q in (2.5, 50, 97.5)],
                                 share_kpeak_gt_kimp=float(np.mean(bs[ok, 1] > bs[ok, 0])*100), n_valid=int(ok.sum()))
print(OUT['month_bootstrap_2C'])
# (5) alternative peak definitions at +2 degC (reference model)
X2 = S.copy(); X2[TFE] = X2[TFE] + 2.0
L2 = np.clip(L0 + ref.predict(X2) - ref.predict(S), 0, None); fac2 = (1 + GAM*(Tc0 + 2 - 25))/(1 + GAM*(Tc0 - 25))
def peakdef(imp, how):
    s = pd.Series(imp*2)
    if how == 'top3_on': return s[on].groupby(mon[on]).apply(lambda v: v.nlargest(3).mean()).mean()
    if how == 'all_hours': return s.groupby(mon).max().mean()
    if how == 'on_max': return s[on].groupby(mon[on]).max().mean()
alt = {}
for how in ['on_max', 'top3_on', 'all_hours']:
    base = peakdef(np.clip(L0 - E0, 0, None), how)
    curve = np.array([peakdef(np.clip(L2 - E0*k*fac2, 0, None), how) for k in KS])
    alt[how] = dict(base=float(base), k_star=kst(base, curve))
OUT['alt_peak_2C'] = alt; print(alt)
# (6) proxy vs billed demand
inv = pd.DataFrame(INVOICES, columns=['month', 'kWh', 'kW', 'THB']).set_index('month')
d = pkm.set_index('month').join(inv).dropna(subset=['kW'])
r = stats.pearsonr(d.Bpk, d.kW)
OUT['billed'] = dict(n=len(d), r=float(r.statistic), p=float(r.pvalue), ratio_median=float((d.Bpk/d.kW).median()),
                     ratio_q=[float((d.Bpk/d.kW).quantile(.25)), float((d.Bpk/d.kW).quantile(.75))])
print(OUT['billed']); print(OUT['marginal'])
json.dump(OUT, open('results_a17.json', 'w'), indent=1, default=float); print('done')
