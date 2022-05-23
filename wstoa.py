import xarray as xr
import numpy as np
import utils 
import dask.array as dka
import warnings
warnings.simplefilter(action='ignore', category=dka.PerformanceWarning)

def Hvelmag(ds,var=('U','V'),name=None):
    da1,da2 = [ds[v] for v in var]
    if name == None:
        name = 'Hvel'
    func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    #return xr.apply_ufunc(func,da1,da2,dask='allowed')
    return ds.assign({name:xr.apply_ufunc(func,da1,da2,dask='allowed')})

def Flux(ds,var,mdim=('level','nj','ni'),name=None):
    if len(var)!=2:
        raise ValueError("var must contain 2 arguments, {:1d} given.".format(len(var)))
    elif type(var[0])!= str:
        raise SyntaxError("var must contain only str objects.")
    if name == None:
        name = (var[0]+var[1]).lower()
    da1,da2 = [ds[v] for v in var]
    return ds.assign({name:(da1-da1.mean(dim=mdim)) * (da2-da2.mean(dim=mdim))})

def WST(arr,Ntht,scales,xydim,var=None,normalize=True):
	tmp = utils.WST(arr,Ntht,scales,xydim,var)
	if normalize==True:
		tmp.normalize()
	return tmp.data