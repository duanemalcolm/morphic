from mesher import Mesh
from fitter import Fit
from fasteval import FastEval

reload_modules = True
if reload_modules:
    import mesher
    reload(mesher)
    from mesher import Mesh
    import fitter
    reload(fitter)
    from fitter import Fit

