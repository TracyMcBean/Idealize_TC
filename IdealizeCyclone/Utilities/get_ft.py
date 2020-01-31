import sys
sys.path.append('/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/modify_init_data/AsymptoticSolver/')

import xarray as xr
import numpy as np

from scipy.interpolate import griddata
from Utilities.coord_func import cart2pol
from AsymptoticSolver import polar_dft, polar_idft, pick_fourier_comp

def get_ft(ds, var_dict, center, r_rad, verbose=True):
    ''' Calculate Fourier Transformation for a given set of variables from a 
    data  set.
        
    Keyword Arguments:
    xrdataset  -- Containing variables
    dictionary -- Containing dictionary of variables which should be idealized
    nparray    -- Lon-Lat positions of center
    float      -- Radius of region to be defined around center
    logical    -- Should run be verbose
    
    Returns: 
    data set with idealized data
    '''
# 1. Polar coordinate transformation
    # This has to be done for every level seperately as it depends on the position of the center.
    nlev = int(nlev)
    lev_start = int(lev_start)

    lon = var_da.clon.values
    lat = var_da.clat.values
    # Extract centerline values. These values are not given for all levels.
    # Because of that, later all indices must be reduced to only account for selected levels.
    x_center = center[:,0]
    y_center = center[:,1]

    # Create grid on which values shall be mapped.
    # Reduce number of points to see point pattern after interpolation
    r_grid = np.linspace(0,r_rad,200).transpose()
    phi_grid = np.linspace(-np.pi,np.pi,200,endpoint=False)

     r_grid_da = xr.DataArray(r_grid, coords=[('r', r_grid)])
    phi_grid_da = xr.DataArray(phi_grid, coords=[('phi', phi_grid)])

    # Calculations for each selected level
    for i in range(50-lev_start,53-lev_start):   # 0, nlev-lev_start+1, 1):
        lev_index = i + lev_start-1
        lev_height = height[lev_index,0]
        print('FT for level: ', lev_index+1, lev_height)

        # Calculate r and phi for single level
        r,phi = cart2pol(lon,lat,center[i,])

        # Unit vector for r and phi
        e_r = np.array([np.cos(phi), np.sin(phi)])
        e_phi= np.array([-np.sin(phi), np.cos(phi)])

# 2. Interpolate data to polar coordinates cells 
        if verbose:
            print('Interpolating data and remapping to polar coordinates...')

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

        # Interpolation 
        # Select single level of variable 
        values = var_da.values[lev_index]
        #print(' This is the level: ', var_da.height[lev-1])
        points = np.asarray([var_da.clon.values[:], var_da.clat.values[:]]).transpose()
        remap_points = np.asarray([x_polar, y_polar]).reshape((2,len(x_polar[0]))).transpose()

        # remap density for circles with constant radius around center
        var_remap = griddata(points, values, remap_points, method='cubic')
        var_remap = var_remap.reshape((len(r_grid),len(phi_grid)))

        # Add polar coordinates as dimensions
        var_polar_da = xr.DataArray(var_remap, coords={ 'r':('r',r_grid), \
                         'phi':('phi', phi_grid), 'x': x_grid , 'y': y_grid }, \
                          dims={'r': r_grid, 'phi':phi_grid })
        var_polar_da = var_polar_da.fillna(0.)

# 3. Fourier transformation
        if verbose:
            print('Starting FT...')

        fvar = polar_dft(var_polar_da, polar_dim='phi')
        fvar_i = polar_idft(fvar, polar_dim='phi')

        # Select 1st mode
        fvar1 = fvar.copy()
        fvar1[0] = 0.
        fvar1[2:] = 0.
        fvar1_i = polar_idft(fvar1)

        # Select only 0 mode
        fvar0 = fvar.copy()
        fvar0[1:] = 0.
        fvar0_i = polar_idft(fvar0)
        # Reduce 0 mode to p_4
        fvar_p4_i = background - fvar0_i


    return ds  
