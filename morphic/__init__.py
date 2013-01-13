from mesher import Mesh
from fitter import Fit
from fasteval import FEMatrix

reload_modules = True
if reload_modules:
    import mesher
    reload(mesher)
    import interpolator
    reload(interpolator)
    from mesher import Mesh
    import fitter
    reload(fitter)
    from fitter import Fit

