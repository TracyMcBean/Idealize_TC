import xarray as xr
import numpy as np
from Utilities.add_var import add_theta, add_u_polar
'''
Create idealized initial data.

28.01.2020, Tracy

'''
# Variables -------------------------------------------------------------------
center_from_file  = True           # If center location should be read from
                                   # array, set to true
center_file       = "./Data/center_fiona.npy"      # Name of file containing center
data_file         = "/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc"
lev_start         = 40             # Level from where the calculations should start
km = 250                           # radius around cyclone
r_earth = 6371                     # earths radius
ft_variables = {'density': True, 'virt pot temp': True, 'pressure':True, \
                'u':True, 'v':True, 'w':True, 'spec humidity':True, 'temperature':True, \
                'spec cloud water': False, 'spec cloud ice': False, \
                'rain mixing ratio': False, 'snow mixing ratio': False }
#------------------------------------------------------------------------------

# 1. Load data

# initial data
ds = xr.open_dataset(data_file)

# cyclone center
if center_from_file:
    try:
        with open(center_file) as cf:
            center = np.load(center_file)
    except FileNotFoundError:
        print('Center file not found. Using minimum pressure.')
        center_from_file = False
else:
    # Calculate center from minimum in case center array is not available.
    pres_da = ds.pres
    
    for l in range(lev_start, nlev+1):
        single_lev = pres_da.isel(height=l)
        center[l-lev_start,0] = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clon.values[0]
        center[l-lev_start,1] = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clat.values[0]

# 2. Preprocess data -----------------------------------------------------------

# Add variables that shall be used in fourier transform
ds = add_theta(ds)
ds = add_u_polar(ds, center)


