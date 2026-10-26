"""Step 01b (alternative to 01a when the raw portal export is not available): rebuild iv.pkl and hosp.pkl
from the released, CT-corrected zone-level series data/zone_30min_series.csv, with the same hospital
aggregation as 01a, and recompute the invoice ratio c_inv (Eq. 1) from the uncorrected import.
Outputs: iv.pkl, hosp.pkl, c_ct.txt, c_inv.txt, tab_invoice_ratio.csv, tab_qc.csv (Table S1, copied)."""
import pandas as pd, numpy as np, holidays, shutil, sys
import os
from paths import DATA
ZF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'zone_30min_series.csv')
C_CT = 1.20                                  # CT-ratio error confirmed on site in all three portal meters
Z = pd.read_csv(ZF, parse_dates=['interval_end_local_time'])
iv = pd.DataFrame({'z': Z.zone.values, 'kWp': Z.zone_kWp.values, 'E': Z.pv_kWh.values, 'L': Z.gross_load_kWh.values,
                   'B': Z.grid_import_kWh.values, 'X': Z.export_kWh.values, 'G': Z.ghi_W_m2.values,
                   'Tamb': Z.ambient_temp_C.values, 'P': Z.pv_power_kW.values, 'st': Z.plant_status.values},
                  index=pd.DatetimeIndex(Z.interval_end_local_time, name='time'))
iv = pd.concat([iv[iv.z == z] for z in sorted(iv.z.unique())])          # zone-major order, as in 01a
# hospital aggregate (all three zones valid); grid flows aggregated on the portal scale, then corrected (as in 01a)
raw = iv.copy(); raw['B'] = raw.B*C_CT; raw['X'] = raw.X*C_CT
w = raw.pivot_table(index='time', columns='z', values=['E', 'L', 'B', 'X', 'G', 'Tamb'])
H = pd.DataFrame(index=w.index)
for c in ['E', 'B', 'X']: H[c] = w[c].sum(axis=1, min_count=3)
H['L'] = w['L'].sum(axis=1, min_count=3)
H = H[['E', 'L', 'B', 'X']]
H['Tamb'] = w['Tamb'].median(axis=1); H['G'] = w['G'].median(axis=1)
H['kWp'] = iv.groupby('z').kWp.first().sum()
th = holidays.TH(years=range(2023, 2027))
H['date'] = H.index.normalize()
H['hol'] = H.date.dt.date.map(lambda x: x in th) | (H.index.dayofweek >= 5)
# invoice ratio on the uncorrected import (Eq. 1)
from pea_invoices import INVOICES
inv = pd.DataFrame(INVOICES, columns=['month', 'kWh', 'kW', 'THB']).set_index('month')
m = H.index.strftime('%Y-%m'); g = H.B.groupby(m); cov = g.count()/(pd.PeriodIndex(g.count().index, freq='M').days_in_month*48)
Bm = (g.sum()/cov)[cov > 0.9]; Bm = Bm[Bm.index.isin(inv.index)]; ratio = Bm/inv.kWh.reindex(Bm.index)
pd.DataFrame({'portal_B_kWh': Bm, 'invoice_kWh': inv.kWh.reindex(Bm.index), 'ratio': ratio, 'coverage': cov.reindex(Bm.index)}).to_csv('tab_invoice_ratio.csv')
open('c_inv.txt', 'w').write(repr(float(ratio.median()))); open('c_ct.txt', 'w').write(repr(C_CT))
print('invoice ratio: n=%d, median=%.4f; after correction %.4f' % (len(ratio), ratio.median(), (ratio/C_CT).median()))
H['B'] = H.B/C_CT; H['X'] = H.X/C_CT; H['L'] = H.E + H.B - H.X
iv.to_pickle('iv.pkl'); H.to_pickle('hosp.pkl'); shutil.copy(os.path.join(DATA, 'tab_qc.csv'), 'tab_qc.csv')
print('hospital intervals', len(H), 'valid load %.2f%%' % (H.L.notna().mean()*100))
