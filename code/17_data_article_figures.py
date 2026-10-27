"""Step 17 (co-submitted Data in Brief article): descriptive figures and tables of the released dataset (1000 dpi PNG + vector PDF)."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pea_invoices import INVOICES
import os
from paths import DATA
D = DATA + os.sep
Z = pd.read_csv(D + 'zone_30min_series.csv', parse_dates=['interval_end_local_time'])
H = pd.read_csv(D + 'hospital_30min_series.csv', parse_dates=['interval_end_local_time']).set_index('interval_end_local_time')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'savefig.dpi': 1000})
C = dict(L='#2a78d6', B='#0b0b0b', E='#eda100', T='#eb6834', G='#1baf7a')
def save(f, n):
    axs = [a for a in f.axes if a.get_label() != '<colorbar>']
    for i, a in enumerate(axs):
        if len(axs) > 1: a.set_title(f'({chr(97+i)})', loc='center', fontsize=10, fontweight='bold')
    f.savefig(f'dib_fig{n}.png', bbox_inches='tight', dpi=1000); f.savefig(f'dib_fig{n}.pdf', bbox_inches='tight'); plt.close(f)
OUT = {}
# ---- Fig 1: data availability per day ----
Z['date'] = Z.interval_end_local_time.dt.normalize()
av = Z.groupby(['zone', 'date']).gross_load_kWh.apply(lambda s: s.notna().mean()*100).unstack(0)
H['date'] = H.index.normalize(); avh = H.groupby('date').gross_load_kWh.apply(lambda s: s.notna().mean()*100)
M = np.vstack([av[1], av[2], av[3], avh]).astype(float)
f, ax = plt.subplots(figsize=(7.4, 1.9))
im = ax.imshow(M, aspect='auto', cmap='Greens', vmin=0, vmax=100, interpolation='nearest',
               extent=[matplotlib.dates.date2num(av.index[0]), matplotlib.dates.date2num(av.index[-1]) + 1, 3.5, -0.5])
ax.set_yticks(range(4), ['Zone 1', 'Zone 2', 'Zone 3', 'Hospital']); ax.xaxis_date()
cb = f.colorbar(im, ax=ax, pad=0.01, fraction=0.03); cb.set_label('Valid intervals (%)', fontsize=8)
save(f, 1)
# ---- Fig 2: carpet plots of gross load and PV (kW) ----
P = H.copy(); P['hh'] = P.index.hour + P.index.minute/60
f, ax = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True, gridspec_kw={'hspace': 0.25})
for a, col, lab, cm in [(ax[0], 'gross_load_kWh', 'Gross load (kW)', 'viridis'), (ax[1], 'pv_kWh', 'PV generation (kW)', 'magma')]:
    W = (P.pivot_table(index='hh', columns='date', values=col, dropna=False)*2).reindex(columns=avh.index)
    im = a.imshow(W.values, aspect='auto', origin='lower', cmap=cm, interpolation='nearest',
                  extent=[matplotlib.dates.date2num(W.columns[0]), matplotlib.dates.date2num(W.columns[-1]) + 1, 0, 24])
    a.set_yticks([0, 6, 12, 18, 24]); a.set_ylabel('Hour of day'); a.xaxis_date()
    c = f.colorbar(im, ax=a, pad=0.01, fraction=0.03); c.set_label(lab, fontsize=8)
save(f, 2)
# ---- Fig 3: monthly invoice check ----
inv = pd.DataFrame(INVOICES, columns=['month', 'kWh', 'kW', 'THB']).set_index('month')
m = H.index.strftime('%Y-%m'); g = H.grid_import_kWh.groupby(m); cov = g.count()/(pd.PeriodIndex(g.count().index, freq='M').days_in_month*48)
Bc = (g.sum()/cov); Br = (H.grid_import_portal_raw_kWh.groupby(m).sum()/cov)
ok = cov[(cov > 0.9) & cov.index.isin(inv.index)].index
T3 = pd.DataFrame({'invoice_MWh': inv.kWh.reindex(ok)/1e3, 'portal_raw_MWh': Br.reindex(ok)/1e3, 'portal_corrected_MWh': Bc.reindex(ok)/1e3, 'coverage': cov.reindex(ok)})
T3['ratio_raw'] = T3.portal_raw_MWh/T3.invoice_MWh; T3['ratio_corr'] = T3.portal_corrected_MWh/T3.invoice_MWh
x = pd.PeriodIndex(T3.index, freq='M').to_timestamp()
f, ax = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw={'wspace': 0.3, 'width_ratios': [1.5, 1]})
ax[0].plot(x, T3.invoice_MWh, color=C['B'], lw=1.6, marker='o', ms=2.5, label='Invoiced energy')
ax[0].plot(x, T3.portal_raw_MWh, color=C['T'], lw=1.2, ls='--', marker='s', ms=2, label='Portal import, uncorrected')
ax[0].plot(x, T3.portal_corrected_MWh, color=C['L'], lw=1.2, marker='^', ms=2, label='Portal import, CT-corrected')
ax[0].set_ylabel('Monthly grid energy (MWh)'); ax[0].legend(fontsize=7, loc='upper left', ncol=1); ax[0].set_ylim(500, 1500)
ax[1].hist(T3.ratio_raw, bins=np.arange(0.90, 1.30, 0.01), color=C['T'], alpha=0.8, label='Uncorrected')
ax[1].hist(T3.ratio_corr, bins=np.arange(0.90, 1.30, 0.01), color=C['L'], alpha=0.8, label='CT-corrected')
ax[1].set_xlabel('Portal import / invoiced energy (–)'); ax[1].set_ylabel('Months'); ax[1].legend(fontsize=7)
f.autofmt_xdate(); save(f, 3)
T3.round(4).to_csv('dib_tab_invoice_check.csv')
OUT['invoice'] = dict(n=len(T3), raw_med=T3.ratio_raw.median(), raw_q=[T3.ratio_raw.quantile(.25), T3.ratio_raw.quantile(.75)],
                      corr_med=T3.ratio_corr.median(), corr_q=[T3.ratio_corr.quantile(.25), T3.ratio_corr.quantile(.75)])
# ---- Fig 4: mean diurnal profiles by season (working days) ----
V = H.dropna(subset=['gross_load_kWh']).copy(); V = V[V.index >= '2023-05-01']
mo = V.index.month; V['season'] = np.select([np.isin(mo, [3, 4, 5]), np.isin(mo, [6, 7, 8, 9, 10])], ['Hot-dry (Mar–May)', 'Wet (Jun–Oct)'], 'Cool-dry (Nov–Feb)')
V['hh'] = V.index.hour + V.index.minute/60; V = V[V.non_working_day == 0]
f, ax = plt.subplots(1, 3, figsize=(7.6, 2.5), sharey=True, gridspec_kw={'wspace': 0.12})
for a, s in zip(ax, ['Cool-dry (Nov–Feb)', 'Hot-dry (Mar–May)', 'Wet (Jun–Oct)']):
    p = V[V.season == s].groupby('hh')[['gross_load_kWh', 'grid_import_kWh', 'pv_kWh']].mean()*2
    a.plot(p.index, p.gross_load_kWh, color=C['L'], lw=1.6, label='Gross load'); a.plot(p.index, p.grid_import_kWh, color=C['B'], lw=1.2, ls='--', label='Grid import')
    a.plot(p.index, p.pv_kWh, color=C['E'], lw=1.6, label='PV'); a.set_xticks([0, 6, 12, 18, 24]); a.set_xlabel('Hour of day'); a.text(0.98, 0.97, s, transform=a.transAxes, ha='right', va='top', fontsize=7.4)
ax[0].set_ylabel('Mean power (kW)'); ax[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 0.88)); ax[0].set_ylim(0, 3100)
save(f, 4)
# ---- tables ----
W = H.dropna(subset=['gross_load_kWh'])
yr = W.groupby(W.index.year).agg(intervals=('gross_load_kWh', 'size'), L=('gross_load_kWh', 'sum'), E=('pv_kWh', 'sum'), B=('grid_import_kWh', 'sum'),
                                 X=('export_kWh', 'sum'), Tmean=('ambient_temp_C', 'mean'))
for c in 'LEBX': yr[c] = yr[c]/1e3
yr['SS'] = yr.E/yr.L*100; yr['Pmax'] = W.groupby(W.index.year).gross_load_kWh.max()*2; yr['Pmean'] = W.groupby(W.index.year).gross_load_kWh.mean()*2
yr.round(2).to_csv('dib_tab_annual.csv'); print(yr.round(1))
zs = Z.groupby('zone').agg(kWp=('zone_kWp', 'first'), valid=('gross_load_kWh', lambda s: s.notna().mean()*100), E=('pv_kWh', 'sum'), L=('gross_load_kWh', 'sum'),
                           B=('grid_import_kWh', 'sum'), X=('export_kWh', 'sum'), Tvalid=('ambient_temp_C', lambda s: s.notna().mean()*100),
                           Gvalid=('ghi_W_m2', lambda s: s.notna().mean()*100))
for c in 'ELBX': zs[c] = zs[c]/1e3
zs.round(2).to_csv('dib_tab_zone.csv'); print(zs.round(2))
OUT['hosp'] = dict(rows=len(H), valid=H.gross_load_kWh.notna().mean()*100, Lmean_kW=(W.gross_load_kWh*2).mean(), Lmax=(W.gross_load_kWh*2).max(),
                   Emax=(W.pv_kWh*2).max(), Tmin=H.ambient_temp_C.min(), Tmax=H.ambient_temp_C.max(), Gmax=H.ghi_W_m2.max(),
                   mar23_mean=(W.loc['2023-03'].gross_load_kWh*2).mean(), med_month=(W.gross_load_kWh.groupby(W.index.to_period('M')).mean()*2).median())
json.dump(OUT, open('dib_stats.json', 'w'), indent=1, default=float); print(OUT)
