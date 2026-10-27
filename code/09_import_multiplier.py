"""Step 09: import-preserving multiplier k*_imp and k*_peak (attribution-independent, Eq. 12) for the
reference model, all six model families and a 20-member ensemble at +2 degC; argument 'raw' repeats the
reference case with the uncorrected portal import (Tables 5, 6, 7, S9, S11).
Outputs: results_a13.json, results_a13_raw.json."""
import sys, json, numpy as np, pandas as pd
RAW = len(sys.argv) > 1 and sys.argv[1] == 'raw'
from paths import code_src
src = code_src('06_multipliers_finegrid.py').split('OUT = {}')[0]
if RAW:
    c = float(open('c_ct.txt').read())
    src = src.replace("S = H.dropna(", "H['L'] = (H.E.fillna(0) + H.B*%r - H.X.fillna(0)*%r).clip(lower=0).where(H.L.notna()); M['L'] = (M.E.fillna(0) + M.B*%r - M.X.fillna(0)*%r).clip(lower=0)\nS = H.dropna(" % (c, c, c, c), 1)
exec(src)
def stress2(m, dT):
    X = S.copy(); X[TFE] = X[TFE] + dT
    L = np.clip(S.L.values + m.predict(X) - m.predict(S), 0, None)
    fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25)); on = S.on.values.astype(bool); mon = S.month.astype(str).values
    r = []
    for k in KS:
        E = S.E.values*k*fac; imp = np.clip(L - E, 0, None)
        r.append((k, imp.sum(), pd.Series(imp[on]*2).groupby(mon[on]).max().mean(), L.sum()))
    return pd.DataFrame(r, columns=['k', 'imp', 'peak', 'L'])
def ks(m, dTs):
    b = stress2(m, 0.0).iloc[0]; out = {}
    for dT in dTs:
        df = stress2(m, dT)
        out[dT] = dict(k_imp=kstar(b.imp, df, 'imp', decreasing=True), k_peak=kstar(b.peak, df, 'peak', decreasing=True),
                       imp_k1_rel=df.iloc[0].imp/b.imp*100 - 100, L_rel=df.iloc[0].L/b.L*100 - 100)
    out['base'] = dict(imp=b.imp, peak=b.peak, L=b.L); return out
yrs = len(set(pk.month))/12; OUT = {}
if RAW:
    m = FAM['GBM, monotone (reference)'](TN0).fit(M); OUT['raw'] = ks(m, [1.0, 2.0, 3.0]); print(OUT)
    json.dump(OUT, open('results_a13_raw.json', 'w'), indent=1, default=float); sys.exit()
for n, f in FAM.items():
    m = f(TN0).fit(M)
    OUT[n] = ks(m, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if 'reference' in n else [1.0, 2.0, 3.0]); print(n, OUT[n], flush=True)
from sklearn.ensemble import HistGradientBoostingRegressor
rng = np.random.default_rng(2026); wk = M.index.to_period('W'); uw = wk.unique(); ens = []
for bi in range(20):
    sw = rng.choice(uw, len(uw)); idx = np.concatenate([np.where(wk == w)[0] for w in sw])
    m = GBM(FEAT, [0,0,0,1,1,1,1,1,0]); m.mod = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=48,
        min_samples_leaf=40, l2_regularization=1.0, monotonic_cst=[0,0,0,1,1,1,1,1,0], categorical_features=[0,1,2], random_state=bi).fit(M.iloc[idx][FEAT], M.iloc[idx].L)
    r = ks(m, [2.0])[2.0]; ens.append(r); print(bi, r, flush=True)
E2 = pd.DataFrame(ens); OUT['ensemble_2C'] = {c: dict(mean=float(E2[c].mean()), lo=float(E2[c].quantile(.025)), hi=float(E2[c].quantile(.975))) for c in E2}
OUT['yrs'] = yrs
json.dump(OUT, open('results_a13.json', 'w'), indent=1, default=float); print(OUT['ensemble_2C']); print('done')
