import numpy as np
from Utilities.coord_func import cart2pol

'''
Calculate variables which shall be added to a given data set
'''

def add_theta(ds):
    """ Add the potential temperature.
    
    Keyword arguments:
    dataset -- contains data to be used
    """
    print('Calculating potential temp...')
    # Extract necessary variables
    # temperature
    try: 
        temp = ds.temp
    except AttributeError: 
        print('Variable temp not found in data set!!')  
    
    # pressure
    try:     
        pres = ds.pres
    except AttributeError:     
        print('Variable pres not found in data set!!') 
    
    # specific humidity
    try: 
        qv = ds.qv
    except AttributeError:
        print('Varialbe qv not found in data set!!')
    
    # set consants
    p_0 = 1000    # reference pressure
    kappa = 0.2854*(1-0.24*qv)  # poisson constant for moist air
   
    theta = temp * (p_0/pres)**kappa
    
    # Add theta to data set
    ds = ds.assign(theta = theta)
    
    return ds

def add_u_polar(ds, center):
    """
    Calculate u_r and u_theta and add them to dataset
    
    Keywords arguments:
    dataset -- contains data for calculation
    center  -- contains location of center on levels
    """
    
    try:
        ds.u
        ds.v
        ds.clon
        ds.clat
        ds.z_ifc
    except AttributeError:
        print('One or several of following variables not found in data set: u, v, clon, clat, z_ifc!!')
    
    print('Calculate u_r and u_theta for following levels: %s to %s' % (ds.z_ifc.values[-len(center),0], ds.z_ifc.values[-1,0]))

    # Number of levels
    nlev = len(ds.height)
    
    # create data array where values shall be replaced
    u_r = ds.u
    u_r.name = 'u_r'
    u_r.attrs['standard_name'] = 'radial_wind'

    u_phi = ds.u
    u_phi.name = 'u_phi'
    u_phi.attrs['standard_name'] = 'tangential wind'
    
    # Use center for possible levels. All other levels will simply be set to zero because they are not necessary.
    for i in range (0,nlev):
        if i < nlev-len(center):
            u_r[0,i] = 0.        
        else:
            # Transform lon-lat coordinates to polar
            ci = i -(nlev-len(center))
            print('This is ci: %s and i: %s' % (ci,i))
            x = ds.clon.values
            y = ds.clat.values
            r,phi = cart2pol(x,y,center[ci,])
            
            # Calculate unit vectors e_r and e_phi
            e_r = np.array([np.cos(phi), np.sin(phi)])
            e_phi= np.array([-np.sin(phi), np.cos(phi)])
 
            u_r[0,i]   = e_r[0]*ds.u[0,i] + e_r[1]*ds.v[0,i]
            u_phi[0,i] = e_phi[0]*ds.u[0,i] + e_phi[1]*ds.v[0,i]

    # Add u_r and u_phi to dataset
    ds = ds.assign(u_r = u_r)
    ds = ds.assign(u_phi = u_phi)
    
    return ds
