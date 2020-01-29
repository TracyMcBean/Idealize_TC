import sys
sys.path.append('/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/modify_init_data/AsymptoticSolver/')

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from Utilities.coord_func import cart2pol
#from AsymptoticSolver.AsymptoticSolver import polar_dft, polar_idft, pick_fourier_comp
from AsymptoticSolver import polar_dft, polar_idft, pick_fourier_comp
from Utilities.ft_var import ft_var

'''
Extract 0 and 1st fourier mode in circle around the center.

Necessary data: Initial data and the array containing the location of
the centerline.
'''

# Variables -------------------------------------------------------------------
center_from_file  = True           # If center location should be read from
                                   # array, set to true
center_file       = "./Data/center_fiona.npy"      # Name of file containing center
data_file         = "/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc"
lev_start         = 35             # Level from where the calculations should start
km = 300                           # radius around cyclone
r_earth = 6371                     # earths radius

#------------------------------------------------------------------------------
### 1. Load data

# Data set with initial data
ds = xr.open_dataset(data_file)

# Number of levels in file (highest index is lowest level -> p-system)
nlev = len(ds.height.values)

# Load cyclone center
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
    center = np.empty([nlev-lev_start,2])
    # Only use levels which are low enough to show explicit signal of cyclone

    for l in range(lev_start, nlev):
        single_lev = pres_da.isel(height=l)
        center[l-lev_start,0] = single_lev.where(single_lev == single_lev.min(),               drop=True).clon.values[0]
        center[l-lev_start,1] = single_lev.where(single_lev == single_lev.min(),               drop=True).clat.values[0]

# Set radius in radian using km (For larger storms value might have to be increased
r_rad = km / r_earth


#------------------------------------------------------------------------------
# Variables for which FT should be done:
calc_rho=False              # Density
calc_pres=False             # Pressure
calc_theta_v=True           # virt. pot. temp
calc_qv=False               # specific humidity
calc_qc=False               # spec. cloud water content
calc_qi=False               # spec. cloud ice content
calc_qr=False               # rain mixing ratio
calc_qs=False               # snow mixing ratio

if calc_rho:
    print('Variable is density.')
    # Get variable without time dimension ([0])
    rho_da = ds.rho[0]
    ft_var(rho_da, center, r_rad, nlev, lev_start, 'Density', 'rho')
    print('Finished density.')

if calc_pres:
    print('Variable is pressure')
    pres_da = ds.pres[0]
    ft_var(pres_da, center, r_rad, nlev, lev_start, 'Pressure', 'pres')
    print('Finished pressure.')

if calc_theta_v:
    print('Variable is virtual potential temperature.')
    theta_v_da = ds.theta_v[0]
    ft_var(theta_v_da, center, r_rad, nlev, lev_start, 'virt_pot_temp', 'theta_v')
    print('Finished virt. pot. temp.')

if calc_qv:
    print('Variable is .')
    qv_da = ds.qv[0]
    ft_var(qv_da, center, r_rad, nlev, lev_start, '', '')
    print('Finished ')

if calc_qc:
    print('Variable is .')
    qc_da = ds.qc[0]
    ft_var(qc_da, center, r_rad, nlev, lev_start, '', '')
    print('Finished ')

if calc_qi:
    print('Variable is .')
    qi_da = ds.qi[0]
    ft_var(qi_da, center, r_rad, nlev, lev_start, '', '')
    print('Finished ')

if calc_qv:
    print('Variable is .')
    qr_da = ds.qr[0]
    ft_var(qr_da, center, r_rad, nlev, lev_start, '', '')
    print('Finished ')

if calc_qs:
    print('Variable is .')
    qs_da = ds.qs[0]
    ft_var(qs_da, center, r_rad, nlev, lev_start, '', '')
    print('Finished ')

#------------------------------------------------------------------------------
# Z. Plots

# Plot centerline:
plot_centerline = False
if plot_centerline:
    # plot for center                                        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.flipud(center[:,0]), np.flipud(center[:,1]), np.flipud(ds.z_ifc.values[35:75,0]))
    plt.title('Centerline of Fiona')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Height at half level')
    plt.show()


# Plot variable
plot_variable = False
if plot_variable:
    fig = plt.figure(figsize=(18,14))
    plt.scatter(x,y,c=var[58-lev_start])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Density on level 58')
    plt.colorbar()
    plt.show()


# Plot variable
plot_variable = False
if plot_variable:
    fig = plt.figure(figsize=(18,14))
    plt.scatter(x,y,c=u_r[58-lev_start])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Magnitude of u_r on level 58')
    plt.colorbar()


