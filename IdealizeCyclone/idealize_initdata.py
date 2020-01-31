import xarray as xr
import numpy as np

from Utilities.add_var import add_theta, add_u_polar
from Utilities.ft_var import ft_var

from matplotlib import pyplot as plt

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
ft_variables = {'density': True, 'virt pot temp': False , 'pressure':True, \
                 'horizontal wind':False, 'w':False, 'spec humidity':False, 'temperature':False, \
                'spec cloud water': False, 'spec cloud ice': False, \
                'rain mixing ratio': False, 'snow mixing ratio': False }
#------------------------------------------------------------------------------

# Set radius in radian using km
r_rad = km / r_earth


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

# 2. Preprocess data -----------------------------------------------------------

# Add variables that shall be used in fourier transform
#ds = add_theta(ds)
#ds = add_u_polar(ds, center)

if ft_variables['density']:
    print('----------------------------------------------------------')
    print('Density')
    ideal_rho_da = ft_var(ds.rho[0], center, r_rad, nlev, lev_start, 'Density', 'rho', height)
    ds.rho[0] = ideal_rho_da

if ft_variables['virt pot temp']:
    print('----------------------------------------------------------')
    print('Virtual potential temperature')
    deal_data_da = ft_var(ds.theta_v[0], center, r_rad, nlev, lev_start, 'virt_pot_temp', 'theta_v', height)
    ds.theta_v[0] = ideal_data_da

if ft_variables['pressure']:
    print('----------------------------------------------------------')
    print('Calculating FT of pressure')
    ideal_data_da = ft_var(ds.pres[0], center, r_rad, nlev, lev_start, 'Pressure', 'pres', height)
    ds.pres[0] = ideal_data_da

if ft_variables['horizontal wind']:
    print('----------------------------------------------------------')
    print('Horizontal wind')

if ft_variables['w']:
    print('----------------------------------------------------------')
    print('Vertical wind')
    ideal_data_da = ft_var(ds.w[0], center, r_rad, nlev, lev_start, 'Wind/w', 'w', height)
    ds.w[0] = ideal_data_da

if ft_variables['temperature']:
    print('----------------------------------------------------------')
    print('Temperature')
    ideal_data_da = ft_var(ds.temp[0], center, r_rad, nlev, lev_start, 'Temperature', 'w', height)
    ds.temp[0] = ideal_data_da

if ft_variables['spec humidity']:
    print('----------------------------------------------------------')
    print('Specific humidity')
    ideal_data_da = ft_var(ds.qv[0], center, r_rad, nlev, lev_start, 'Humidity/spec_humidity', 'qv', height)
    ds.qv[0] = ideal_data_da

if ft_variables['spec cloud water']:
    print('----------------------------------------------------------')
    print('Specific cloud water content')
    ideal_data_da = ft_var(ds.qc[0], center, r_rad, nlev, lev_start,'Humidity/spec_cwc', 'qc', height)
    ds.qc[0] = ideal_data_da


if ft_variables['spec cloud ice']:
    print('----------------------------------------------------------')
    print('Specific cloud ice content')
    ideal_data_da = ft_var(ds.qi[0], center, r_rad, nlev, lev_start,'Humidity/spec_cic', 'qi', height)
    ds.qc[0] = ideal_data_da

if ft_variables['rain mixing ratio']:
    print('----------------------------------------------------------')
    print('Rain mixing ratio')
    ideal_data_da = ft_var(ds.qr[0], center, r_rad, nlev, lev_start,'Humidity/rain_mr', 'qr', height)
    ds.qr[0] = ideal_data_da

if ft_variables['snow mixing ratio']:
    print('----------------------------------------------------------')
    print('Snow mixing ratio')
    ideal_data_da = ft_var(ds.qs[0], center, r_rad, nlev, lev_start, 'Humidity/snow_mr', 'qs', height)
    ds.qs[0] = ideal_data_da


print(ds)

plt.figure()
plt.tripcolor(ds.clon, ds.clat, ds.pres.isel(time = 0, height=49))
plt.colorbar()
plt.show()
