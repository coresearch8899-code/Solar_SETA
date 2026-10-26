"""Step 13: IPCC AR6 warming levels mapped onto the stress test (Table S9) and synthesis figure fig10 (Fig. S5).
Outputs: tab_ipcc_envelope.csv, results_ipcc.json, fig10.png."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
G = pd.read_csv('stress_grid.csv'); X = json.load(open('results_extra.json'))
# IPCC AR6 Synthesis Report (2023) best-estimate global warming vs 1850-1900; 2011-2020 = 1.1 degC
ipcc = [('SSP2-4.5', '2041–2060', 2.0), ('SSP5-8.5', '2041–2060', 2.4), ('SSP2-4.5', '2081–2100', 2.7), ('SSP5-8.5', '2081–2100', 4.4)]
dts = sorted(G.dT.unique()); kS = [X['kstar'][str(float(d))]['k_SHHR'] for d in dts]
kP = [X['kstar'][str(float(d))]['k_peak'] if X['kstar'][str(float(d))]['k_peak'] is not None else np.nan for d in dts]
A13 = json.load(open('results_a13.json'))['GBM, monotone (reference)']   # from 09_import_multiplier.py
kI = [1.0 if d == 0 else A13[str(float(d))]['k_imp'] for d in dts]
g1 = G[G.k == 1].set_index('dT'); rows = []
for ssp, per, glob in ipcc:
    d = round(glob - 1.1, 2); inr = d <= max(dts)
    rows.append(dict(scenario=ssp, period=per, global_warming=glob, dT_vs_2011_2020=d,
                     L=np.interp(d, dts, g1.L_GWh.loc[dts]) if inr else np.nan, SHHR_hp=np.interp(d, dts, g1.SHHR.loc[dts]) if inr else np.nan,
                     peak=np.interp(d, dts, g1.peak_net.loc[dts]) if inr else np.nan, kI=np.interp(d, dts, kI) if inr else np.nan, kS=np.interp(d, dts, kS) if inr else np.nan,
                     kP=np.interp(d, dts, kP) if d <= 2.5 else np.nan))
T = pd.DataFrame(rows); T.to_csv('tab_ipcc_envelope.csv', index=False); print(T.round(2).to_string())
json.dump(T.to_dict('records'), open('results_ipcc.json', 'w'), indent=1, default=float)

# ---------------- Fig. 9 conceptual synthesis ----------------
C = dict(b='#2a78d6', o='#eb6834', a='#1baf7a', y='#eda100', v='#4a3aa7', g='#52514e', m='#e87ba4')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
f, ax = plt.subplots(figsize=(9.2, 4.6)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
def box(x, y, w, h, t, c, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=c+'22', ec=c, lw=1.4))
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=fs)
def arr(x0, y0, x1, y1): ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=C['g'], lw=1.2))
box(0.1, 4.2, 2.1, 1.5, 'Ambient temperature\n(T, 24-h, 72-h means)\nT$_b$ ≈ 20.0 °C', C['b'])
box(2.6, 4.2, 2.3, 1.5, 'Model-attributed\ntemperature-driven load\n≈ 31% of demand\n(28–33%, 6 models)', C['o'])
box(5.3, 4.2, 2.3, 1.5, 'Thermal lag\nheat load peaks ≈ 2.5 h\nafter PV; ⅓ at night', C['v'])
box(8.0, 4.2, 3.9, 1.5, 'Solar-heat hedge (explanatory;\nattribution-dependent)\nHEOR 67%  |  SHHR ≈ 23–25%\n(bounds 3–48%); η$_{hedge}$ = 0.35', C['a'])
box(8.0, 2.2, 3.9, 1.5, 'Peak displacement (measured)\n30-min on-peak proxy −697 kW\npeaks ≥ 15:00: 17% → 64%\ndemand cost ≈ 9% of\ntariff-equivalent saving', C['y'])
box(4.3, 2.2, 3.3, 1.5, 'Warming stress test\nΔT = +2 °C (within the IPCC AR6\nlate-century warming range)', C['m'])
box(0.1, 0.2, 5.6, 1.5, 'PV-only preservation multipliers (site-specific)\nk*$_{imp}$ ≈ 1.6×   vs   k*$_{peak}$ ≈ 1.9–3.2×\n→ energy-peak divergence (attribution-independent;\nall 6 model families; uncorrected-data case)', C['o'])
box(6.3, 0.2, 5.6, 1.5, 'Complementary late-afternoon flexibility\nstorage, chilled-water/ice storage, pre-cooling,\nwest-facing PV (screening at +2 °C:\nk = 3 + 5 MWh → peak 3% below present)', C['a'])
arr(2.2, 4.95, 2.6, 4.95); arr(4.9, 4.95, 5.3, 4.95); arr(7.6, 4.95, 8.0, 4.95); arr(9.95, 4.2, 9.95, 3.7)
arr(8.0, 2.95, 7.6, 2.95); arr(5.95, 2.2, 3.2, 1.7); arr(5.95, 2.2, 8.9, 1.7)
f.savefig('fig10.png', bbox_inches='tight', dpi=1000); plt.close(f); print('fig9 done')
