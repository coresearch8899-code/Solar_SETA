"""Step 10: sensitivity of k*_imp and k*_peak at +2 degC to the PV temperature coefficient (Table 6).
Output: results_a13_gamma.json."""
import json
from paths import code_src
src = code_src('09_import_multiplier.py').split('yrs = len(set(pk.month))/12')[0]
exec(src)
m = FAM['GBM, monotone (reference)'](TN0).fit(M); out = {}
for g in [-0.003, -0.004, -0.0045]:
    GAM = g; out[g] = ks(m, [2.0])[2.0]; print(g, out[g], flush=True)
json.dump(out, open('results_a13_gamma.json', 'w'), indent=1, default=float)
