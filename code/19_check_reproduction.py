"""Step 19: compare a rerun with the reference outputs: python check_reproduction.py reference_outputs . results.json results_rev.json ..."""
import json, sys, math
A, B = sys.argv[1], sys.argv[2]
def flat(d, p=''):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items(): o.update(flat(v, f'{p}{k}.'))
    elif isinstance(d, list):
        for i, v in enumerate(d): o.update(flat(v, f'{p}{i}.'))
    elif isinstance(d, (int, float)) and not isinstance(d, bool): o[p[:-1]] = float(d)
    return o
for f in sys.argv[3:]:
    try: a = flat(json.load(open(f'{A}/{f}'))); b = flat(json.load(open(f'{B}/{f}')))
    except FileNotFoundError as e: print(f, 'missing', e); continue
    keys = set(a) & set(b); worst = []
    for k in keys:
        x, y = a[k], b[k]
        if math.isnan(x) and math.isnan(y): continue
        r = abs(x - y)/max(abs(x), 1e-9); worst.append((r, k, x, y))
    worst.sort(reverse=True)
    print(f'{f}: {len(keys)} values, missing {len(set(a)^set(b))}, max rel diff {worst[0][0]:.2e}' if worst else f)
    for w in worst[:4]:
        if w[0] > 1e-6: print('   ', w)
