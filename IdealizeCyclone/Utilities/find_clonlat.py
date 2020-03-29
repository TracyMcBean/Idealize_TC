import numpy as np
import math
import xarray as xr
from numba import jit

@jit(parallel=True)
def get_clonlat(fg, single_lev, p_env, r):
   '''
   Get the location of the center point on a single level, given a first guess and a data set.
   r is radius of circle that should be selected.
   '''
   
   # Extract first guess lon and lat
   clon_fg = fg[0]
   clat_fg = fg[1]
   
   # Cut out quadratic box containing all cells in possible circle region
   clon_lim_east = clon_fg + r
   clon_lim_west = clon_fg - r
   clat_lim_north= clat_fg + r
   clat_lim_south= clat_fg - r

   # extract box containing all possible cells
   single_lev = single_lev.where(single_lev['clon'] < clon_lim_east, drop=True)
   single_lev = single_lev.where(single_lev['clon'] > clon_lim_west, drop=True)
   single_lev = single_lev.where(single_lev['clat'] < clat_lim_north, drop=True)
   single_lev = single_lev.where(single_lev['clat'] > clat_lim_south, drop=True)
   
   # Get radial distance of each cell to first guess.

   dist_array = np.empty([len(single_lev.ncells)])
   dist_array = np.sqrt((single_lev_u.clon.values - center_loc[0])**2 \
                 + (single_lev_u.clat.values - center_loc[1])**2)
    
   #for i in range(len(single_lev.ncells)):
   #   dist_array[i]= math.sqrt((single_lev.clon.values[i] -clon_fg)**2 \
   #               + (single_lev.clat.values[i] - clat_fg)**2)

   # Create new dataset combining pres and dist data.
   circle_sel = xr.Dataset({'pres': single_lev, 'dist': (('ncells'), dist_array)})
   # Extract region in circle
   circle_sel = circle_sel.where(circle_sel['dist'] <= r, drop=True)

   # Calculate average over all cells in circle region
   #p_env = np.mean(circle_sel.pres.values)
   # I am using p_max because otherwise I end up with lots of negative values and therefore get wrong values
   #p_max = circle_sel.pres.values.max()
   p_prime = p_env - circle_sel.pres.values
   #p_prime = p_max - circle_sel.pres.values

   clon_new = np.sum(circle_sel.clon.values * p_prime)/np.sum(p_prime)
   clat_new = np.sum(circle_sel.clat.values * p_prime)/np.sum(p_prime)

   clon_new = circle_sel.clon.values[np.argmin(np.abs(circle_sel.clon.values \
              - clon_new))]
   clat_new = circle_sel.clat.values[np.argmin(np.abs(circle_sel.clat.values \
              - clat_new))]
   
   if clat_fg != clat_new or clon_fg != clon_new:
      # Recursively find the center
      print('starting new calculation for center')
      print(clon_new, clat_new)
      get_clonlat([clon_new, clat_new], single_lev, p_env, r)
   else:
      print('Found: ', clon_new, clat_new) 
      return (clon_new, clat_new)

   return (clon_new, clat_new)
