import sys
sys.path.append('/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/modify_init_data/AsymptoticSolver/')

import numpy as np
import xarray as xr
import math
from Utilities.coord_func import cart2pol
from AsymptoticSolver import polar_dft, polar_idft, pick_fourier_comp
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

''' Compare theoretical tangential wind derived from equation 4.14a in Päschke et al (2012) with the tangential wind calculated using.
'''

# Get necessary data
center = np.load('./Data/center_fiona.npy')  
ds = xr.open_dataset('/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc')

r_rad =300/6371 # chosen radius 
lev_start = 35
height = ds.z_ifc

# Calculate coriolis parameter (Use center positions that make sense)
mean_center_lat = center[2:,1].mean()*188/np.pi
omega=7.27*10**(-5)  # earths angular velocity
f=2*omega*math.sin(mean_center_lat)

# Get density, pressure and hor. wind field
rho = ds.rho[0]
pres= ds.pres[0]
u = ds.u[0]
v = ds.v[0]

#Create arrays for circumferential and radial wind
u_phi  = np.empty([len(ds.height),len(ds.ncells.values)])

lon = rho.clon.values
lat = rho.clat.values
x = ds.clon.values
y = ds.clat.values
x_center = center[:,0]
y_center = center[:,1]

# Create grid on which values shall be mapped.
r_grid = np.linspace(0,r_rad,1000).transpose()
phi_grid = np.linspace(-np.pi,np.pi,1000,endpoint=False)

r_grid_da = xr.DataArray(r_grid, coords=[('r', r_grid)])
phi_grid_da = xr.DataArray(phi_grid, coords=[('phi', phi_grid)])

# Calculations for each selected level
for i in range(50-lev_start,51-lev_start):
#for i in range(0, nlev-lev_start+1, 1):
    lev_index = i + lev_start-1
    lev_height = height[lev_index,0]
    print('FT for level: ', lev_index+1, lev_height.values)

    # Calculate r and phi for single level
    r,phi = cart2pol(lon,lat,center[i,])

    # Unit vector for r and phi
    e_r = np.array([np.cos(phi), np.sin(phi)])
    e_phi= np.array([-np.sin(phi), np.cos(phi)])

    u_phi[i] = e_phi[0]*ds.u.values[0,i] + e_phi[1]*ds.v.values[0,i]
    print(u_phi.max(), u_phi.min())

# 2. Interpolate data to polar coordinates cells 
    print('Interpolating data and mapping to polar coordinates...')

    x_grid = x_center[i] + r_grid_da*np.cos(phi_grid_da)
    y_grid = y_center[i] + r_grid_da*np.sin(phi_grid_da)

    # Create new lon and lat positions using polar coordinates 
    # (necessary to have all points for interpolation method):
    x_polar = [x_center[i] + r_grid[j]*np.cos(phi_grid) \
              for j in range(len(r_grid))]
    x_polar = np.asarray(x_polar).reshape((1,len(r_grid)*len(r_grid)))
    y_polar = [y_center[i] + r_grid[j]*np.sin(phi_grid) \
              for j in range(len(r_grid))]
    y_polar = np.asarray(y_polar).reshape((1,len(r_grid)*len(r_grid)))


    rho_values = rho.values[lev_index]
    pres_values = pres.values[lev_index]    
    u_phi_values = u_phi[i]    

    lonlat_points = np.asarray([lon[:],lat[:]]).transpose()
    polar_points = np.asarray([x_polar, y_polar]).reshape((2,len(x_polar[0]))).transpose()

    # remap variables for circles with constant radius around center
    rho_remap = griddata(lonlat_points, rho_values, polar_points, method='cubic')
    rho_remap = rho_remap.reshape((len(r_grid),len(phi_grid)))

    pres_remap = griddata(lonlat_points, pres_values, polar_points, method='cubic')
    pres_remap = pres_remap.reshape((len(r_grid),len(phi_grid)))
    
    u_phi_remap = griddata(lonlat_points, u_phi_values, polar_points, method='cubic')
    u_phi_remap = u_phi_remap.reshape((len(r_grid),len(phi_grid)))

    # Add polar coordinates as dimensions
    rho_polar_da = xr.DataArray(rho_remap, coords={ 'r':('r',r_grid), \
                     'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                      dims={'r': r_grid, 'phi':phi_grid })
    rho_polar_da = rho_polar_da.fillna(0.)

    pres_polar_da = xr.DataArray(pres_remap, coords={ 'r':('r',r_grid), \
                     'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                      dims={'r': r_grid, 'phi':phi_grid })
    pres_polar_da = pres_polar_da.fillna(0.)

    u_phi_polar_da = xr.DataArray(u_phi_remap, coords={ 'r':('r',r_grid), \
                         'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                          dims={'r': r_grid, 'phi':phi_grid })
    u_phi_polar_da = u_phi_polar_da.fillna(0.)


    rho_background = rho_values[lev_index].mean()
    pres_background = pres_values[lev_index].mean()

    frho = polar_dft(rho_polar_da, polar_dim='phi')
    frho_i = polar_idft(frho, polar_dim='phi')
    fpres = polar_dft(pres_polar_da, polar_dim='phi')
    fpres_i = polar_idft(fpres, polar_dim='phi')

    # Select fourier mode 0
    frho0 = frho
    frho0[1:]=0.
    frho0_i = xr.ufuncs.real(polar_idft(frho0, polar_dim='phi'))
    fpres0 = fpres
    fpres0[1:]=0.
    fpres0_i = xr.ufuncs.real(polar_idft(fpres0, polar_dim='phi'))

    fpres_p4_i = fpres0_i - pres_background
    print(fpres_p4_i)    

    delta_p4 = np.copy(fpres_p4_i)
    delta_p4[:,-1] = 0.
    # gradient function verwenden (von xarray)
    for p in range(0, len(phi_grid)):
        delta_p4[p, 0:-1] = np.asarray(fpres_p4_i[p, 0:-1]) - np.asarray(fpres_p4_i[p, 1:])
    delta_r = np.copy(delta_p4)
    delta_r[:,-1] = 0.  
    for p in range(0, len(phi_grid)):  
    delta_r[p, 0:-1] = np.asarray(fpres_p4_i.r[0:-1]) - np.asarray(fpres_p4_i.r[1:]) 
         
    rho_background = rho_values[lev_index].mean()
    print(delta_p4) 
    # Calculate theoretical u_phi
    u_phi_1 = 1/2*(-f + np.sqrt(f**2 - 4/rho_background * delta_p4/delta_r))

  #  u_phi_2 = 1/2*(-f - np.sqrt(f**2 - 4/rho_background * delta_p4))

    #print('u_phi_1', u_phi_1)
    #print('u_phi_2', u_phi_2)    
    u_phi_1 = xr.DataArray(u_phi_1, coords={ 'r':('r',r_grid), \
                     'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                      dims={'r': r_grid, 'phi':phi_grid })
    u_phi_2 = xr.DataArray(u_phi_2, coords={ 'r':('r',r_grid), \
                     'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                      dims={'r': r_grid, 'phi':phi_grid })
    u_phi_1 = u_phi_1.fillna(0.) 
    #print('u_phi_1', u_phi_1)
    #print('u_phi_2', u_phi_2)    
    
    # plot theoretical u_phi against actual u_phi
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(121)
    cs = ax.pcolor(u_phi_polar_da.x, u_phi_polar_da.y, u_phi_polar_da)
    ax.title.set_text('u_phi from wind (%s m)' %  np.int(lev_height))
    cb = plt.colorbar(cs, ax=ax)

    ax = fig.add_subplot(122)
    ax.axes.get_yaxis().set_visible(False)
    #cs = ax.pcolor(fvar0_i.x, fvar0_i.y, xr.ufuncs.real(fvar0_i))
    cs = ax.pcolor(u_phi_1.x, u_phi_1.y, u_phi_1)
    ax.title.set_text('u_phi_1')
    cb = plt.colorbar(cs, ax=ax)
'''
    ax = fig.add_subplot(111)
    ax.axes.get_yaxis().set_visible(False)
    #cs = ax.pcolor(fvar0_i.x, fvar0_i.y, xr.ufuncs.real(fvar0_i))
    cs = ax.pcolor(u_phi_2.x, u_phi_2.y, u_phi_2)
    ax.title.set_text('u_phi_2')
    cb = plt.colorbar(cs, ax=ax)
'''   
 plt.show()

