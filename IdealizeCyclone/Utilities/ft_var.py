#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../../AsymptoticSolver/')

import xarray as xr
import numpy as np
#from numba import jit
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from Utilities.coord_func import cart2pol
# Choose either smooth or discrete blending
from Utilities.coord_func import sel_blending_smooth as sel_blending
from AsymptoticSolver import polar_dft, polar_idft, pick_fourier_comp


#@jit   #(parallel=True)
def ft_var(var_da, center, r_rad, nlev, lev_start, var_name, var_nshort, height, verbose = True, create_plot = False, r_earth=6371):
    '''
    Convert a given variable from lon-lat into polar coordinates.
    Extract 0 and 1st fourier mode in circle around the center save plots if required.
    Create blending zone on border of idealized data and save blended and idealized data.
    Return data array with variable which has been idealized around center.

    Keyword arguments:
    dataarray -- Variable for which FT should be done
    array     -- Contains centerline
    float     -- radius in radians for circle to be selected around centerline
    int       -- number of levels for which variable is given
    int       -- level from which calculations should start
    string    -- Name of variable (path to file in plots folder)
    string    -- Name of variable as given in dataarray
    array     -- Cotaining height in unit (m/km/p etc)
    boolean   -- True if verbose
    boolean   -- True if plots should be created and saved
    int       -- Earths radius (Constant but depending on source)
    '''
# 1. Polar coordinate transformation
    # Save original value data for blending later
    orig_da = var_da
    
    # This has to be done for every level seperately as it depends on the position of the center.
    nlev = int(nlev)
    lev_start = int(lev_start)
  
     
    # Get background necessary for analysing perturbation
    background = np.empty([nlev])
    for i in range(0, nlev):
        background[i] = var_da.values[i].mean()
    
    # limits of quadratic region to select
    # include some buffer because center varies in each level
    lonlat_box = {'lon_up':-0.57,'lon_down':-0.68, 'lat_up': 0.17, 'lat_down': 0.30}

    # To conserve the indexing the cellID must be added as coordinate variable
    var_da = var_da.assign_coords({'ncells': var_da.ncells.values}) 
 
    # Select region of interest
    var_da = var_da.where(var_da['clon'] < lonlat_box['lon_up'], drop=True)
    var_da = var_da.where(var_da['clon'] > lonlat_box['lon_down'], drop=True)
    var_da = var_da.where(var_da['clat'] > lonlat_box['lat_up'], drop=True)
    var_da = var_da.where(var_da['clat'] < lonlat_box['lat_down'], drop=True)
    
    # The cellID is necessary for the blending later
    cellID = var_da.ncells.values
    
    lon = var_da.clon.values
    lat = var_da.clat.values
    # Extract centerline values. These values are not given for all levels.
    # Because of that, later all indices must reduced to only account for selected levels.
    x_center = center[:,0]
    y_center = center[:,1]

    # Create grid on which values shall be mapped.
    # Reduce number of points to see point pattern after interpolation
    r_grid = np.linspace(0,r_rad,1000).transpose()
    phi_grid = np.linspace(-np.pi,np.pi,1000,endpoint=False)

    r_grid_da = xr.DataArray(r_grid, coords=[('r', r_grid)])
    phi_grid_da = xr.DataArray(phi_grid, coords=[('phi', phi_grid)])

    # Calculations for each selected level
    for i in range(66,66+1):  
    #for i in range(lev_start, nlev+1, 1):
        center_index = i- (nlev - len(center))-1
        lev_index = i-1
        lev_height = height[lev_index,0]
        print('FT for level: ', lev_index+1, lev_height)
        print('center index:', center_index)
 
        # Calculate r and phi for single level
        r,phi = cart2pol(lon,lat,center[center_index,])

        # Unit vector for r and phi
        e_r = np.array([np.cos(phi), np.sin(phi)])
        e_phi= np.array([-np.sin(phi), np.cos(phi)])
   
# 2. Interpolate data to polar coordinates cells 
        if verbose:
            print('Interpolating data and mapping to polar coordinates...')   
        
        x_grid = x_center[center_index] + r_grid_da*np.cos(phi_grid_da)
        y_grid = y_center[center_index] + r_grid_da*np.sin(phi_grid_da)

        # Create new lon and lat positions using polar coordinates 
        # (necessary to have all points for interpolation method):
        x_polar = [x_center[center_index] + r_grid[j]*np.cos(phi_grid) \
                  for j in range(len(r_grid))]
        x_polar = np.asarray(x_polar).reshape((1,len(r_grid)*len(r_grid)))
        y_polar = [y_center[center_index] + r_grid[j]*np.sin(phi_grid) \
                  for j in range(len(r_grid))]
        y_polar = np.asarray(y_polar).reshape((1,len(r_grid)*len(r_grid)))

        # plt.title('Points on which variable should be mapped')
        # plt.scatter(x_polar, y_polar)                   

        # Interpolation (griddata() is slow!)
        # Select single level of variable 
        values = var_da.values[lev_index]
        #print(' This is the level: ', var_da.height[lev-1])
        lonlat_points = np.asarray([var_da.clon.values[:], var_da.clat.values[:]]).transpose()
        polar_points = np.asarray([x_polar, y_polar]).reshape((2,len(x_polar[0]))).transpose()
        
        # remap variables for circles with constant radius around center
        var_remap = griddata(lonlat_points, values, polar_points, method='cubic')
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

        # Select fourier mode 0 and 1
        if var_nshort == 'u_phi' or var_nshort == 'u_r': 
            # For horizontal winds I have to use k=-1 as well as 0 and 1 mode
            print('detected horizontal wind therefore using k=-1,0,1')
            fvar01 = fvar
            fvar[2:-1] = 0.
            fvar01_i = xr.ufuncs.real(polar_idft(fvar01, polar_dim='phi'))
        else:
            # For all other variables select only Fourier mode 0 and 1
            fvar01 = fvar
            fvar01[2:]=0.
            fvar01_i = xr.ufuncs.real(polar_idft(fvar01, polar_dim='phi')) # only real part      
        
        if create_plot:
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
            print('background:', background[lev_index])
            fvar_p4_i = fvar0_i - background[lev_index]
    
# 4. Transform idealized data back to lon lat grid.
        if verbose:
           print('Mapping idealized data to lonlat...')
        
        values = np.asarray(fvar01_i).reshape(len(polar_points[:,0]))
        
        # Create correct mapping points for each cell
        new_x_points= np.asarray(fvar01_i.x.values).reshape(1,len(fvar01_i.x.values[0])*len(fvar01_i.x.values[0]))
        new_y_points= np.asarray(fvar01_i.y.values).reshape(1,len(fvar01_i.y.values[0])*len(fvar01_i.y.values[0]))
        polar_points = np.asarray([new_x_points, new_y_points]).reshape((2,len(new_x_points[0]))).transpose() 
        
        # Remap data
        var_remap = griddata(polar_points, values, lonlat_points, method='cubic')
       
        var_ideal_da = xr.DataArray(var_remap,  coords={ 'r': ('ncells', r), \
                         'phi': ('ncells', phi), 'clon': ('ncells', lon) , \
                         'clat': ('ncells', lat), 'cellID': ('ncells', cellID)}, \
                          dims={'ncells': var_da.ncells })
      
        # Still sanity check for fourier modes remapping
        if False:
            if verbose:
                print('Plotting idealized data that should be remapped')
            fig = plt.figure(figsize=(9,3))
            ax = fig.add_subplot(121)
            cs = ax.pcolor(fvar01_i.x, fvar01_i.y, fvar01_i)
            ax.title.set_text('%s (%s m)' % (var_name, np.int(lev_height)))
            cb = plt.colorbar(cs, ax=ax)
            plot_region_da = var_ideal_da.where(var_ideal_da.r < (r_rad-0.001) , drop = True)
            
            ax = fig.add_subplot(122)
            cs = ax.tripcolor(plot_region_da.clon, plot_region_da.clat, plot_region_da.values)
            ax.title.set_text('ideal %s (%s m)' % (var_name, np.int(lev_height)))
            cb = plt.colorbar(cs, ax=ax)
            plt.show()

# 5. Replace original data with idealized data using a blending zone to avoid jumps 
        if verbose:
            print('Blending idealized and original data...')
 
        # Calculate limits where logistic function should start and end.       
        km50_rad = 50/r_earth
        # The limit is set a bit shorter than r_rad to avoid including nan data
        blend_lim = [(r_rad - km50_rad), r_rad-1e-07] 
        
        # Extract blending zone (ring from r_rad-50km to r_rad)
        blend_zone = var_ideal_da.where(var_ideal_da.r > blend_lim[0], drop=True)  
        blend_zone = blend_zone.where(blend_zone.r < blend_lim[1], drop=True)
       
        cell_r = blend_zone.r[blend_zone.ncells.values]
        # Get idealized and original value
        val_ideal = blend_zone.values[blend_zone.ncells.values]
        val_orig  = orig_da.values[lev_index, blend_zone.cellID[blend_zone.ncells.values]]
        
        val_cell = sel_blending(cell_r, r_rad, val_ideal, val_orig)
        if np.any(np.isnan(val_cell)):
            print('These are the nan val_ideal, val_orig, cell_r:')
            nan_ids = np.where(np.isnan(val_cell)==True)
            print(val_ideal[nan_ids], val_orig[nan_ids], cell_r[nan_ids])

        orig_da.values[lev_index, blend_zone.cellID] = val_cell

        # Replace all values in cells with radius distance less than blending start
        inner_zone = var_ideal_da.where(var_ideal_da.r <= blend_lim[0], drop=True)
        orig_da.values[lev_index,inner_zone.cellID[inner_zone.ncells.values]] = inner_zone.values[inner_zone.ncells.values]
        #for cell_index in inner_zone.ncells.values:
        #    orig_da.values[lev_index,inner_zone.cellID[cell_index]] = inner_zone.values[cell_index]

        
# Creating plot
        if create_plot:
            if verbose:
                print('Creating and saving plot of the fourier modes')
            
            fig = plt.figure(figsize=(9,3))
            ax = fig.add_subplot(131)
            cs = ax.pcolor(fvar_i.x, fvar_i.y, xr.ufuncs.real(fvar_i))
            ax.title.set_text('%s (%s m)' % (var_name, np.int(lev_height)))
            cb = plt.colorbar(cs, ax=ax) 
    
            ax = fig.add_subplot(132)
            ax.axes.get_yaxis().set_visible(False)
            #cs = ax.pcolor(fvar0_i.x, fvar0_i.y, xr.ufuncs.real(fvar0_i))
            cs = ax.pcolor(fvar0_i.x, fvar0_i.y, xr.ufuncs.real(fvar_p4_i))
            ax.title.set_text('Fourier mode 0 (p_4)')
            cb = plt.colorbar(cs, ax=ax)
    
            ax = fig.add_subplot(133)
            ax.axes.get_yaxis().set_visible(False)
            cs = ax.pcolor(fvar1_i.x, fvar1_i.y, xr.ufuncs.real(fvar1_i))
            ax.title.set_text('Fourier mode 1')
            cb = plt.colorbar(cs, ax=ax)
        
            # Save image for each level
            plt.savefig('/home/bekthkis/Plots/Fiona/%s/updated_%s_ft_lev_%s.png' % (var_name, var_nshort, np.int(lev_height) ))
            plt.close()
    
    print('Finished ft_var()')
    return orig_da
