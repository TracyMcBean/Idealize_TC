#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/modify_init_data/AsymptoticSolver/')

import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utilities.coord_func import cart2pol, sel_blending
from AsymptoticSolver import polar_dft, polar_idft, pick_fourier_comp

'''
Extract 0 and 1st fourier mode in circle around the center for a given variable and save plots.


'''

def ft_var(var_da, center, r_rad, nlev, lev_start, var_name, var_nshort, height, verbose = True, create_plot = False, r_earth=6371):
# 1. Polar coordinate transformation
    # This has to be done for every level seperately as it depends on the position of the center.
    nlev = int(nlev)
    lev_start = int(lev_start)

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
    for i in range(64-lev_start,nlev-lev_start+1):  
    #for i in range(0, nlev-lev_start+1, 1):
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

        # plt.title('Points on which variable should be mapped')
        # plt.scatter(x_polar, y_polar)                   

        # Interpolation 
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
        
        background = var_da.values[lev_index].mean() 

        fvar = polar_dft(var_polar_da, polar_dim='phi')
        fvar_i = polar_idft(fvar, polar_dim='phi')

        # Select fourier mode 0 and 1
        fvar01 = fvar
        fvar01[2:]=0.
        fvar01_i = xr.ufuncs.real(polar_idft(fvar01, polar_dim='phi')) # only real part      
        # Sanity check for fourier modes
        if False:
            fig = plt.figure(figsize=(9,3))
            ax = fig.add_subplot(121)
            cs = ax.pcolor(fvar01_i.x, fvar01_i.y, fvar01_i)
            ax.title.set_text('%s (%s m)' % (var_name, np.int(lev_height)))
            cb = plt.colorbar(cs, ax=ax)

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
            fvar_p4_i = fvar0_i - background
    
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
            plot_region_da = var_ideal_da.where(var_ideal_da.r < (r_rad-0.001) , drop = True) 
            ax = fig.add_subplot(122)
            cs = ax.tripcolor(plot_region_da.clon, plot_region_da.clat, plot_region_da)
            ax.title.set_text('%s (%s m)' % (var_name, np.int(lev_height)))
            cb = plt.colorbar(cs, ax=ax)
            plt.show()
        
        # Calculate limits where logistic function should start and end.       
        km50_rad = 50/r_earth
        blend_lim = [(r_rad - km50_rad), r_rad] 
        
        # Extract blending zone (ring from r_rad-50km to r_rad)
        blend_zone = var_ideal_da.where(var_ideal_da.r > blend_lim[0], drop=True)  
        blend_zone = blend_zone.where(blend_zone.r < blend_lim[1], drop=True)
       
        # Overwrite the values of the original data array with the idealized data using a logistic function (see sel_blendind()) 
        for cell_index in blend_zone.ncells.values:
            cell_r = blend_zone.r[cell_index]
            val_ideal = blend_zone.values[cell_index]
            val_orig  = var_da.values[lev_index, blend_zone.cellID[cell_index]]
            
            val_cell = sel_blending(cell_r, r_rad, val_ideal, val_orig)

            var_da.values[lev_index, blend_zone.cellID[cell_index]] = val_cell       

        # Replace all values in cells with radius distance less than blending start
        inner_zone = var_ideal_da.where(var_ideal_da.r <= blend_lim[0], drop=True)
        for cell_index in inner_zone.ncells.values:
            var_da.values[lev_index,inner_zone.cellID[cell_index]] = inner_zone.values[cell_index]

        
# Creating plot
        if create_plot:
            if verbose:
                print('Creating plot...')
            
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
            plt.savefig('/home/bekthkis/Plots/Fiona/%s/%s_ft_lev_%s.png' % (var_name, var_nshort, np.int(lev_height) ))
    
            plt.close()
    print('Finished ft_var()')
    return var_da
