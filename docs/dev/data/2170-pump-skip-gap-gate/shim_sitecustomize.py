# Harness-only (#2170 panel): this host has no `ipopt` executable; gdplib's mod_hens and
# kaibel builders call SolverFactory("ipopt") for setup solves. Route that name to the
# discopt Pyomo plugin. Applied identically to both panel arms.
import importlib.abc, sys

def _patch():
    import pyomo.environ  # noqa: F401
    import pyomo.opt.base.solvers as S
    import discopt.pyomo  # noqa: F401  (registers "discopt")
    real = S.SolverFactory.__class__.__call__
    orig = S.SolverFactory
    class _F:
        def __getattr__(self, k):
            return getattr(orig, k)
        def __call__(self, name=None, **kw):
            if name == "ipopt":
                name = "discopt"
            return orig(name, **kw)
    f = _F()
    S.SolverFactory = f
    import pyomo.environ as pe
    pe.SolverFactory = f
    import pyomo.opt as po
    po.SolverFactory = f

class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "gdplib" and not getattr(self, "done", False):
            self.done = True
            _patch()
        return None
sys.meta_path.insert(0, _Hook())
