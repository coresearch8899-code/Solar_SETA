import sys, matplotlib
matplotlib.use('Agg')
import matplotlib.figure as mf
_orig = mf.Figure.savefig
def sv(self, fname, *a, **k):
    _orig(self, fname, *a, **k)
    if isinstance(fname, str) and fname.endswith('.png'):
        k2 = dict(k); k2.pop('dpi', None); _orig(self, fname[:-4] + '.pdf', *a, **k2)
mf.Figure.savefig = sv
script = sys.argv[1]; sys.argv = sys.argv[1:]
exec(compile(open(script).read(), script, 'exec'), {'__name__': '__main__'})
