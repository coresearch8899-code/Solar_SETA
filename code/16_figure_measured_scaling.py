"""Step 16 (Fig. 3; Table 3): figure of the measured response to PV scaling at the present climate (no load model) and of the
PV output at the time of the monthly gross and net peaks (fig_scale)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = pd.read_pickle('hosp.pkl'); pk = pd.read_csv('peaks_monthly.csv'); KWP = 2513.2
S = H[H.index.strftime('%Y-%m').isin(set(pk.month))].dropna(subset=['L', 'E'])
# on-peak flag as defined in step 02 (09:00-22:00 on working days)
S2 = pd.read_pickle('hosp2.pkl'); S2 = S2[S2.index.isin(S.index)]
on = S2.on.reindex(S.index).fillna(False).values.astype(bool); mon = S.index.strftime('%Y-%m').values
L, E = S.L.values, S.E.values; ks = np.round(np.arange(0, 4.001, 0.1), 2); rows = []
for k in ks:
    imp = np.clip(L - k*E, 0, None)
    rows.append((k, imp.sum(), pd.Series(imp[on]*2).groupby(mon[on]).max().mean()))
T = pd.DataFrame(rows, columns=['k', 'imp', 'peak'])
T['imp_red'] = (1 - T.imp/T.imp.iloc[0])*100; T['peak_red'] = (1 - T.peak/T.peak.iloc[0])*100
T['m_imp'] = -T.imp.diff()/T.k.diff(); T['m_peak'] = -T.peak.diff()/T.k.diff()
i1 = T.index[T.k == 1.0][0]; first_imp = (T.imp.iloc[0] - T.imp.iloc[i1]); first_peak = (T.peak.iloc[0] - T.peak.iloc[i1])
T['m_imp_rel'] = T.m_imp/first_imp*100; T['m_peak_rel'] = T.m_peak/first_peak*100
T.to_csv('tab_measured_scaling_fine.csv', index=False)
C = dict(b='#2a78d6', o='#eb6834', g='#52514e', y='#eda100')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
f, ax = plt.subplots(1, 3, figsize=(10.2, 3.1), gridspec_kw={'wspace': 0.36, 'width_ratios': [1, 1, 0.8]})
ax[0].plot(T.k, T.imp_red, color=C['b'], lw=2, label='Annual grid import')
ax[0].plot(T.k, T.peak_red, color=C['o'], lw=2, label='Mean monthly 30-min net peak')
ax[0].axvline(1, color=C['g'], ls=':', lw=1); ax[0].text(1.05, 2, 'present\nfleet', fontsize=7, color=C['g'])
ax[0].set_xlabel('PV scale factor k (× 2.51 MWp)'); ax[0].set_ylabel('Reduction relative to no PV (%)'); ax[0].legend(fontsize=7.2, loc='upper left')
x = T.k.iloc[1:] - 0.05
ax[1].plot(x, T.m_imp_rel.iloc[1:], color=C['b'], lw=2, label='Annual grid import')
ax[1].plot(x, T.m_peak_rel.iloc[1:], color=C['o'], lw=2, label='Net peak')
ax[1].axvline(1, color=C['g'], ls=':', lw=1); ax[1].set_ylim(0, 200)
ax[1].set_xlabel('PV scale factor k (× 2.51 MWp)'); ax[1].set_ylabel('Marginal reduction per added kWp\n(% of the average for the present fleet)')
ax[1].legend(fontsize=7.2, loc='upper right')
EtL = pk.E_at_Lpk.values/KWP*100; EtB = H.E.reindex(pd.to_datetime(pk.tB)).values*2/KWP*100
bp = ax[2].boxplot([EtL, EtB], widths=0.5, patch_artist=True, medianprops=dict(color='#0b0b0b'))
for p, c in zip(bp['boxes'], [C['b'], C['o']]): p.set_facecolor(c + '55'); p.set_edgecolor(c)
rng = np.random.default_rng(1)
for j, v in enumerate([EtL, EtB]): ax[2].scatter(j + 1 + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color='#0b0b0b', alpha=0.6, zorder=3)
ax[2].set_xticks([1, 2], ['At gross\npeak', 'At net\npeak']); ax[2].set_ylabel('PV output (% of rated capacity)')
for i, a in enumerate(ax): a.set_title(f'({chr(97+i)})', loc='center', fontsize=10, fontweight='bold')
f.savefig('fig_scale.png', bbox_inches='tight', dpi=1000); print(T[T.k.isin([0, 1, 1.5, 2, 3, 4])].round(2).to_string())
