from scipy.interpolate import griddata
import xarray as xr
import numpy as np

def get_wind_bg(ds):

    u_da_3d = ds.u[0]
    lon_min = ds.clon.values.min()  
    lon_max = ds.clon.values.max()  
    lat_min = ds.clat.values.min()  
    lat_max = ds.clat.values.max()  

    lon_reg = np.linspace(lon_min, lon_max, 1000)
    lat_reg = np.linspace(lat_min, lat_max, 1000)

    lon_remap = np.empty(len(lon_reg)*len(lon_reg)) 
    lat_remap = np.empty(len(lat_reg)*len(lat_reg)) 

    c = 0 
    l_grid = len(lon_reg) 
    for i in range(0,l_grid):       
        lon_remap[i*l_grid:i*l_grid+l_grid] = lon_reg       
        lat_remap[i*l_grid+c:i*l_grid+l_grid] = lat_reg[0:l_grid-c]   
        lat_remap[i*l_grid:i*l_grid+c] = lat_reg[l_grid-c:]       
        c += 1        
                    
 
    # Interpolate wind data onto regular grid
    points = np.asarray([ds.u.clon.values[:], ds.u.clat.values[:]]).transpose()
    remap_points = np.asarray([lon_remap, lat_remap]).transpose()

    # Select single level of u 
    values = u_da_3d[0]
    # remap density for circles with constant radius around center
    u_remap = griddata(points, values, remap_points, method='cubic')
    u_remap = u_remap.reshape((len(lat_reg),len(lon_reg)))

    get_Q(u_remap) 


