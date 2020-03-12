import sys
sys.path.append('../../AsymptoticSolver/')

from scipy.interpolate import griddata
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utilities.coord_func import cart2pol
from AsymptoticSolver import polar_dft, polar_idft, pick_fourier_comp

'''
Extract 0 and 1st fourier mode of horizontal wind components in circle around the center. To do this u and v are used to calculate u_phi and u_r and those are used for the FT. Then they are mapped back.

Necessary data: Initial data and the array containing the location of
the centerline.
'''

# Variables -------------------------------------------------------------------
center_from_file  = True           # If center location should be read from
                                   # array, set to true
center_file       = "./Data/center_fiona.npy"      # Name of file containing center
data_file         = "../../../init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc"
lev_start         = 35             # Level from where the calculations should start
km = 250                           # radius around cyclone
r_earth = 6371                     # earths radius
plot_p4 = False                     # Create plot for fourier mode 0 only p4 part
#------------------------------------------------------------------------------
# 1. Load data
# Load initial data
ds = xr.open_dataset(data_file)

# Number of levels in file (highest index is lowest level -> p-system)
nlev = len(ds.height.values)
height = ds.z_ifc.values

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
    
    for l in range(lev_start, nlev+1):
        single_lev = pres_da.isel(height=l)
        center[l-lev_start,0] = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clon.values[0]
        center[l-lev_start,1] = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clat.values[0]

# Set radius in radian using km (For larger storms value might have to be increased
r_rad = km / r_earth

#------------------------------------------------------------------------------
# 2. polar coordinate transformation
# This has to be done for every level seperately

x = ds.clon.values
y = ds.clat.values
# Extract centerline values. These values are not given for all levels.
# Because of that later all indices must reduced to only account for selected levels.
x_center = center[:,0]
y_center = center[:,1]

# Create grid on which values shall be mapped.
# Reduce number of points to see point pattern after interpolation
r_grid = np.linspace(0,r_rad,1000).transpose()
phi_grid = np.linspace(-np.pi,np.pi,1000,endpoint=False)

r_grid_da = xr.DataArray(r_grid, coords=[('r', r_grid)])
phi_grid_da = xr.DataArray(phi_grid, coords=[('phi', phi_grid)])

# Select wind
#u = ds.u.values[0,(lev_start-1):nlev]
#v = ds.v.values[0,(lev_start-1):nlev]

#Create arrays for circumferential and radial wind
u_r    = np.empty([nlev-lev_start+1,len(ds.ncells.values)])
u_phi  = np.empty([nlev-lev_start+1,len(ds.ncells.values)])
mag_u_r  = np.empty([nlev-lev_start+1,len(ds.ncells.values)])
mag_u_phi= np.empty([nlev-lev_start+1,len(ds.ncells.values)])


print('Polar coordinate transformation...')

for i in range(60-lev_start,61-lev_start):#(0, nlev-lev_start): #nlev-lev_start+1, 1):
    lev_index = i + lev_start-1
    print('FT for level: ', lev_index+1)

    # Calculate r and phi for single level (here level 35)
    r,phi = cart2pol(x,y,center[i,])
    
    # Unit vector for r and phi
    e_r = np.array([np.cos(phi), np.sin(phi)])
    e_phi= np.array([-np.sin(phi), np.cos(phi)])

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

    # Calculate radial and circumferential wind 
    u_r[i]   = e_r[0]*ds.u.values[0,i] + e_r[1]*ds.v.values[0,i]
    u_phi[i] = e_phi[0]*ds.u.values[0,i] + e_phi[1]*ds.v.values[0,i]
    
    # Magnitude of velocities
    mag_u_r[i] = np.abs(u_r[i])
    mag_u_phi[i] = np.abs(u_phi[i])

    # Interpolation
    # Create points of data and points that should be mapped on to 
    points = np.asarray([ds.u.clon.values[:], ds.u.clat.values[:]]).transpose()
    remap_points = np.asarray([x_polar, y_polar]).reshape((2,len(x_polar[0]))).transpose()

    # Select single level of u_r 
    values = u_r[i]
    # remap density for circles with constant radius around center
    u_r_remap = griddata(points, values, remap_points, method='cubic')
    u_r_remap = u_r_remap.reshape((len(r_grid),len(phi_grid)))

    # Add polar coordinates as dimensions
    u_r_polar_da = xr.DataArray(u_r_remap, coords={ 'r':('r',r_grid), \
                         'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                          dims={'r': r_grid, 'phi':phi_grid })
    u_r_polar_da = u_r_polar_da.fillna(0.)

    # Select single level of u_phi 
    values = u_phi[i]
    # remap density for circles with constant radius around center
    u_phi_remap = griddata(points, values, remap_points, method='cubic')
    u_phi_remap = u_phi_remap.reshape((len(r_grid),len(phi_grid)))

    # Add polar coordinates as dimensions
    u_phi_polar_da = xr.DataArray(u_phi_remap, coords={ 'r':('r',r_grid), \
                         'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                          dims={'r': r_grid, 'phi':phi_grid })
    u_phi_polar_da = u_phi_polar_da.fillna(0.)

# 3. Fourier transformation
    background_ur = u_r_polar_da.values.mean()

    fur = polar_dft(u_r_polar_da, polar_dim='phi')
    fur_i = polar_idft(fur, polar_dim='phi')

    # Select 1st mode
    fur1 = fur.copy()
    fur1[0] = 0.
    fur1[2:] = 0.
    fur1_i = polar_idft(fur1)

    # Select only 0 mode
    fur0 = fur.copy()
    fur0[1:] = 0.
    fur0_i = polar_idft(fur0)

    fur_p4_i = background_ur - fur0_i

    print('Creating plot...')
    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(131)
    cs = ax.pcolor(fur_i.x, fur_i.y, xr.ufuncs.real(fur_i))
    ax.title.set_text('radial wind (%s m)' % np.int(height[lev_index,0]))
    cb = plt.colorbar(cs, ax=ax)

    ax = fig.add_subplot(132)
    ax.axes.get_yaxis().set_visible(False)
    if plot_p4:
        cs = ax.pcolor(fur_p4_i.x, fur_p4_i.y, xr.ufuncs.real(fur_p4_i))
        ax.title.set_text('Fourier mode 0 (p4)')
    else:
        cs = ax.pcolor(fur0_i.x, fur0_i.y, xr.ufuncs.real(fur0_i))
        ax.title.set_text('Fourier mode 0')
    cb = plt.colorbar(cs, ax=ax)

    ax = fig.add_subplot(133)
    ax.axes.get_yaxis().set_visible(False)
    cs = ax.pcolor(fur1_i.x, fur1_i.y, xr.ufuncs.real(fur1_i))
    ax.title.set_text('Fourier mode 1')
    cb = plt.colorbar(cs, ax=ax)

    # Save image for each level
    plt.savefig('/home/bekthkis/Fiona2016/Plots/Wind/u_r/mag_ur_ft_lev_%s.png' % np.int(height[lev_index,0]))

    plt.close()

# 3. Fourier transformation
    background_uphi = u_phi_polar_da.values.mean()

    fuphi = polar_dft(u_phi_polar_da, polar_dim='phi')
    fuphi_i = polar_idft(fuphi, polar_dim='phi')

    # Select 1st mode
    fuphi1 = fuphi.copy()
    fuphi1[0] = 0.
    fuphi1[2:] = 0.
    fuphi1_i = polar_idft(fuphi1)

    # Select only 0 mode
    fuphi0 = fuphi.copy()
    fuphi0[1:] = 0.
    fuphi0_i = polar_idft(fuphi0)
    
    # Get p4 from mode 0
    fuphi_p4_i = background_uphi - fuphi0_i

    print('Creating plot...')
    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(131)
    cs = ax.pcolor(fuphi_i.x, fuphi_i.y, xr.ufuncs.real(fuphi_i))
    ax.title.set_text('tan. wind ( %s m)' % np.int(height[lev_index, 0]))
    cb = plt.colorbar(cs, ax=ax)
    
    ax = fig.add_subplot(132)
    ax.axes.get_yaxis().set_visible(False)
    if plot_p4:    
        cs = ax.pcolor(fuphi_p4_i.x, fuphi_p4_i.y, xr.ufuncs.real(fuphi_p4_i))
        ax.title.set_text('Fourier mode 0 (p_4)')
    else:
        cs = ax.pcolor(fuphi0_i.x, fuphi0_i.y, xr.ufuncs.real(fuphi0_i))
        ax.title.set_text('Fourier mode 0')
    cb = plt.colorbar(cs, ax=ax)

    ax = fig.add_subplot(133)
    ax.axes.get_yaxis().set_visible(False)
    cs = ax.pcolor(fuphi1_i.x, fuphi1_i.y, xr.ufuncs.real(fuphi1_i))
    ax.title.set_text('Fourier mode 1')
    cb = plt.colorbar(cs, ax=ax)

    # Save image for each level
    plt.savefig('/home/bekthkis/Fiona2016/Plots/Wind/u_phi/mag_uphi_ft_lev_%s.png' % np.int(height[lev_index,0]))

    plt.close()




# Old stuff----------------------------------------------------------------
'''
# The following is for a single level
# 3. Interpolate data to polar coordinates cells 
# Create grid on which values shall be mapped.       
r_coord = np.linspace(0,2,100000)
phi_coord = np.linspace(-np.pi,np.pi,640,endpoint=False)
r_coord = xr.DataArray(r_coord, coords=[('r',r_coord)])                
phi_coord = xr.DataArray(phi_coord, coords=[('phi', phi_coord)])       

# Add phi and r locations as dimension to the dataset:
phi_da = xr.DataArray(phi, coords=[('phi',phi)])
r_da   = xr.DataArray(r, coords=[('r', r)])

u_da_singlelev = xr.DataArray(u_da[0], coords=[('phi',phi)], dims=['phi'])
#assign_coords doesnt work for some reason. Haven't found a solution yet. 

# New x and y positions:
x_polar = x_center[0] + r_coord*np.cos(phi)
y_polar = y_center[0] + r_coord*np.sin(phi)


# Z. Plot stuff
plot_centerline = False
if plot_centerline:
    # plot for center                                        
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')    
    ax.plot(np.flipud(center[:,0]), np.flipud(center[:,1]), np.flipud(ds.z_ifc.values[0:41,0]))
    plt.title('Centerline of Fiona')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Height at half level')
    plt.show()

plot_uriable = False
if plot_uriable:
    fig = plt.figure()
    plt.scatter(x,y,c=uriable1d)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('')
    plt.colorbar()
'''
print('End.')
