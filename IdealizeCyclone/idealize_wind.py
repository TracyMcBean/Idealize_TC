import sys
import xarray as xr
import numpy as np
from Utilities.add_var import add_u_polar, get_uv_from_polar, get_bg_wind
from Utilities.ft_var import ft_var

from matplotlib import pyplot as plt

'''
Create idealized initial data using given initial data. Tested for Fiona dataset.

28.01.2020, Tracy

'''
# Variables -------------------------------------------------------------------
center_from_file  = True           # If center location should be read from
                                   # array, set to true
center_file       = "./Data/center_fiona.npy"      # Name of file containing center
data_file         = "../../../init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc"
data_out_file     = "../../../init_data/wind_idealized.nc"
save_ds           = False         # Save data set containing idealized data
lev_start         = 45            # Level from where the calculations should start
km = 250                           # radius around cyclone
r_earth = 6371                     # earths radius
ft_variables = {'horizontal wind':True, 'w':True}
#------------------------------------------------------------------------------

# Set radius in radian using km
r_rad = km / r_earth
print('r_rad', r_rad)
# initial data
ds = xr.open_dataset(data_file)

# Number of levels in file (highest index is lowest level -> p-system)
nlev = len(ds.height.values)
height = ds.z_ifc.values

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

# 2. Idealize data -----------------------------------------------------------

if ft_variables['horizontal wind']:
    print('----------------------------------------------------------')
    print('Horizontal wind')   
    
    with_bg = False
    if with_bg:
        print('Computing background wind...') 
        # Get background wind
        bg_wind = get_bg_wind(ds) 
 
        # Get wind without background
        nobg_u = ds.u[0].copy()
        nobg_v = ds.v[0].copy()

        for i in range(0,nlev):
            nobg_u[i] = nobg_u[i] - bg_wind[0][i]
            nobg_v[i] = nobg_v[i] - bg_wind[1][i]

    # Add hor. wind in polar coordinates
    if with_bg:
        ds = add_u_polar(ds, nobg_u, nobg_v, center) 
    else:
        ds = add_u_polar(ds, ds.u[0], ds.v[0], center)

    ideal_u_phi = ft_var(ds.u_phi[0], center, r_rad, nlev, lev_start, 'Wind/u_phi', 'u_phi', height, create_plot=False) 
    ideal_u_r = ft_var(ds.u_r[0], center, r_rad, nlev, lev_start, 'Wind/u_r', 'u_r', height, create_plot=False)
    
    #print('This is clon:')
    #print(ds.clon)
    # Calculate u and v based on u_phi and u_r
    ds = get_uv_from_polar(ds, center, add_bg=False, bg_wind=[0])
    #print('This is clon after:')
    #print(ds.clon)

if ft_variables['w']:
    print('----------------------------------------------------------')
    print('Vertical wind')
    ideal_data_da = ft_var(ds.w[0], center, r_rad, nlev, lev_start, 'Wind/w', 'w', height, create_plot=False)
    ds.w[0] = ideal_data_da

#print(ds.clon)

# 3. Save idealized data set
save_ds=True
if save_ds:
    ds.to_netcdf(data_out_file, mode = 'w', format='NETCDF4')

if False:
    plt.figure()
    plt.tripcolor(ds.clon, ds.clat, ds.u.isel(time = 0, height=69))
    plt.colorbar()
    plt.show()
'''
plt.figure()
plt.tripcolor(ds.clon, ds.clat, ds.theta_v.isel(time = 0, height=70))
plt.colorbar()
plt.show()
'''
