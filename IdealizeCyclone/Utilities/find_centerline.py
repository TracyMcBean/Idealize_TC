import xarray as xr
import numpy as np
import math
from matplotlib import pyplot as plt 
#from guppy import hpy; h=hpy()
from find_clonlat import get_clonlat
from multiprocessing import Pool

''' 
This script schould calculate the centerline using the pressure centroid method.  

See:  
L. T. NGUYEN, J. MOLINARI and D. THOMAS (2014):  
Evaluation of Tropical Cyclone Center Identification Methods in Numerical Models

Note on the side
km to radian
rad = dist in km / Earth radius in km
100 km ~ 0.015696 rad  
''' 

# Variables ===================================================================

r_rad = 200/6371          # Using radius of 200 km calculate the radius in radian
lev = 75                  # Set number of levels (starting from 1, lowest 
                          # z-height is highest level)
lev_start = 35            # Level from where the calculations should begin 
center = np.empty([41,2]) # Array containing center coordinates for each level
lonlat_box = {'lon_up':-0.57,'lon_down':-0.68, 'lat_up': 0.17, 'lat_down': 0.30}
                          # limits of quadratic region to select
                          #  include some buffer because center varies in each level
filename = "center_fiona" # Name of file where center array should be saved
save = True               # Should center array be saved in to "filename"
#==============================================================================

# Read in data
pres_ds = xr.open_dataset('../Data/pres_data.nc')

# extract region of fiona 
pres_ds = pres_ds.where(pres_ds['clon'] < lonlat_box['lon_up'], drop=True)
pres_ds = pres_ds.where(pres_ds['clon'] > lonlat_box['lon_down'], drop=True)
pres_ds = pres_ds.where(pres_ds['clat'] > lonlat_box['lat_up'], drop=True)
pres_ds = fpres_ds.where(pres_ds['clat'] < lonlat_box['lat_down'], drop=True)

#print(h.heap())

# check if correct region is selected
show_region = False
if show_region:
    print("Creating plot of region...")
    plt.tripcolor(pres_ds.clon, pres_ds.clat, pres_ds.pres.isel(time=0, height=-1))
    plt.show()
else:
    print("Not creating plot.")

# The center must be found on each level
while lev >= lev_start :
    print( 'Finding center on lev:', lev)

    single_lev = pres_ds.pres.isel(height=lev-1)
    # 1. First guess:
    clon_fg = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clon.values[0]
    clat_fg = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clat.values[0]

    center[lev-lev_start,] = get_clonlat([clon_fg, clat_fg], single_lev, r_rad)
    lev -= 1
    print(center)

if save:
    print(" Writing center array into: ", filename)
    np.save(filename,center)

'''
me trying to parallelize this... fail.   

def get_center(lev_set):
    center_array = np.empty([len(lev_set),2])
    for l in lev_set: 
        print( 'Finding center on lev:', l)

        single_lev = pres_ds.pres.isel(height=l)
        # 1. First guess:
        clon_fg = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clon.values[0]
        clat_fg = single_lev.where(single_lev == single_lev.min(), \
              drop=True).clat.values[0]

        center_array[l-35,] = get_clonlat([clon_fg, clat_fg], single_lev, r_rad)
    print(center_array)

pool = Pool(4)
lev_set = np.asarray(range(35,74))
pool.map(get_center, lev_set.tolist())

'''


