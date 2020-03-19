import numpy as np
import math
import xarray as xr
from numba import jit

@jit(parallel=True)
def get_lonlat(fg, single_lev, p_env, r):
    '''
    Get the location of the center point on a single level, given a first guess and a data set.
    r is radius of circle that should be selected.
    '''
    # Extract first guess lon and lat
    lon_fg = fg[0]
    lat_fg = fg[1]
    
    r = r 
    # Cut out quadratic box containing all cells in possible circle region
    lon_lim_east = lon_fg + r
    lon_lim_west = lon_fg - r
    lat_lim_north= lat_fg + r
    lat_lim_south= lat_fg - r

    # extract box containing all possible cells
    single_lev = single_lev.where(single_lev['lon'] < lon_lim_east, drop=True)
    single_lev = single_lev.where(single_lev['lon'] > lon_lim_west, drop=True)
    single_lev = single_lev.where(single_lev['lat'] < lat_lim_north, drop=True)
    single_lev = single_lev.where(single_lev['lat'] > lat_lim_south, drop=True)
   
    # Get radial distance of each cell to first guess.
    # Following version of distance calculation gave wrong values:
    #dist_test = scipy.spatial.distance.cdist(np.asarray([single_lev.clon.values, single_lev.clat.values]).reshape((len(lon_dist),2)), np.asarray([clon_fg, clat_fg]).reshape((1,2))) 
    lon_array = np.empty([len(single_lev.lon)*len(single_lev.lat)]) 
    lat_array = np.empty([len(single_lev.lon)*len(single_lev.lat)])
    pres_array = np.empty([len(single_lev.lon)*len(single_lev.lat)])
    dist_array = np.empty([len(single_lev.lon)*len(single_lev.lat)])
    c = 0

    for lon_i in range(len(single_lev.lon)):
        for lat_i in range(len(single_lev.lat)):  
            dist_array[c]= math.sqrt((single_lev.lon.values[lon_i] -lon_fg)**2 \
                  + (single_lev.lat.values[lat_i] - lat_fg)**2)
            lon_array[c] = single_lev.lon.values[lon_i]
            lat_array[c] = single_lev.lat.values[lat_i]
            pres_array[c] = single_lev.pres.values[lat_i, lon_i]
            c += 1


   
    # Create new dataset combining pres,lon,lat and dist data.
    circle_sel = xr.Dataset({'pres': pres_array, 'dist': dist_array, 'lon': lon_array, 'lat': lat_array})
    # Extract region in circle
    circle_sel = circle_sel.where(circle_sel['dist'] <= r, drop=True)
 

    print(circle_sel.pres.values)
    # Calculate average over all cells in circle region
    #p_env = np.mean(circle_sel.pres.values)
    # I am using p_max because otherwise I end up with lots of negative values and therefore get wrong values
    #p_max = circle_sel.pres.values.max()
    p_prime = p_env - circle_sel.pres.values

    lon_new = np.sum(circle_sel.lon.values * p_prime)/np.sum(p_prime)
    lat_new = np.sum(circle_sel.lat.values * p_prime)/np.sum(p_prime)

    lon_new = circle_sel.lon.values[np.argmin(np.abs(circle_sel.lon.values \
              - lon_new))]
    lat_new = circle_sel.lat.values[np.argmin(np.abs(circle_sel.lat.values \
              - lat_new))]
   
    if lat_fg != lat_new or lon_fg != lon_new:
        # Recursively find the center
        print('starting new calculation for center')
        print(lon_new, lat_new)
        get_lonlat([lon_new, lat_new], single_lev, p_env, r)
    else:
        print('Found: ', lon_new, lat_new) 
        return (lon_new, lat_new)

    return (lon_new, lat_new)
