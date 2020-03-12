import numpy as np
from Utilities.coord_func import cart2pol

'''
Functions related to handling variables that should be added or modified from the data set. 

Includes: add_theta, add_u_polar, get_uv_from_polar, get_bg_wind
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

def add_u_polar(ds, u, v, center):
    """
    Calculate u_r and u_theta and add them to dataset
    
    Keyword arguments:
    dataset -- contains data for calculation
    dataarray -- Contains u-component of wind (no time dim)
    dataarray -- Contains v-component of wind (no time dim)
    center  -- contains location of center on levels
    """
    
    try:
        #ds.u
        #ds.v
        ds.clon
        ds.clat
        ds.z_ifc
    except AttributeError:
        print('One or several of following variables not found in data set: u, v, clon, clat, z_ifc!!')
    
    print('Calculate u_r and u_phi for following levels: %s to %s' % (ds.z_ifc.values[-len(center),0], ds.z_ifc.values[-1,0]))

    # Number of levels
    nlev = len(ds.height)
    
    # create data array where values shall be replaced
    u_r =ds.u.copy()
    u_r.name = 'u_r'
    u_r.attrs['standard_name'] = 'radial_wind'

    u_phi = ds.u.copy()
    u_phi.name = 'u_phi'
    u_phi.attrs['standard_name'] = 'tangential_wind'
    
    # Use center for possible levels. All other levels will simply be set to zero because they are not necessary.
    for i in range (0,nlev):
        if i < nlev-len(center):
            u_r[0,i] = 0.        
        else:
            # Transform lon-lat coordinates to polar
            ci = i -(nlev-len(center))
            x = ds.clon.values
            y = ds.clat.values
            r,phi = cart2pol(x,y,center[ci,])
            
            # Calculate unit vectors e_r and e_phi
            e_r = np.array([np.cos(phi), np.sin(phi)])
            e_phi= np.array([-np.sin(phi), np.cos(phi)])
 
            u_r[0,i]   = e_r[0]*u[i] + e_r[1]*v[i]
            u_phi[0,i] = e_phi[0]*u[i] + e_phi[1]*v[i]

    # Add u_r and u_phi to dataset
    ds = ds.assign(u_r = u_r)
    ds = ds.assign(u_phi = u_phi)
    
    return ds

def get_uv_from_polar(ds, center, add_bg=False, bg_wind=[0] ):
    """
    Calculate u and v based on given polar coordinates.
    
    Keyword arguments:
    dataset -- contains data for calculation
    array  -- contains 2d location of center on levels
    bool   -- True if background wind should be added
    array  -- Background wind for u and v each level 
    """
    
    try:
        ds.u_phi
        ds.u_r
        ds.u
        ds.v
        ds.clon
        ds.clat
        ds.z_ifc
    except AttributeError:
        print('One or several of following variables not found in data set: u_phi, u_r, u, v, clon, clat, z_ifc!!')
    
    print('Calculate v and u for following levels: %s to %s' % (ds.z_ifc.values[-len(center),0], ds.z_ifc.values[-1,0]))

    # Number of levels for these parameters
    nlev = len(ds.height)
    
    u = ds.u.copy()
    v = ds.v.copy()
    
    # Only replace levels of ds.u and ds.v where we have calculated u_phi and u_r.
    for i in range (0,nlev):
        if i >= nlev-len(center):
            # Transform lon-lat coordinates to polar
            ci = i -(nlev-len(center))
            x = ds.clon.values
            y = ds.clat.values
            r,phi = cart2pol(x,y,center[ci,])
            
            # Calculate unit vectors e_r and e_phi
            e_r = np.array([np.cos(phi), np.sin(phi)])
            e_phi= np.array([-np.sin(phi), np.cos(phi)])
            
            # Calculate u and v 
            u[0,i] = ds.u_r[0,i] * e_r[0] +  ds.u_phi[0,i] * e_phi[0]
            v[0,i] = ds.u_r[0,i] * e_r[1] +  ds.u_phi[0,i] * e_phi[1]

            # Now blend this data too!
            # I have r given
        
        # Add background wind back again if required
        if add_bg:    
            print('Wind before adding bg:')
            print(u[0,i])
            print('Background:')
            print(bg_wind[0][i])       
            print("Adding background wind...")
            #print(u[0,i], bg_wind[0][i])
            u[0,i] = u[0,i] + bg_wind[0][i]
            v[0,i] = v[0,i] + bg_wind[1][i]
            print('After adding bg:')
            print(u[0,i])

    # Exchange values of u and v 
    ds = ds.assign(u = u)
    ds = ds.assign(v = v)
   
    return ds

def get_uv(u_phi, u_r, center):

    u = u_phi.copy()
    v = u_phi.copy()
    nlev = len(u_phi.height)
    
    for i in range (0,nlev):
        if i >= nlev-len(center):
            # Transform lon-lat coordinates to polar
            ci = i -(nlev-len(center))
            x = ds.clon.values
            y = ds.clat.values
            r,phi = cart2pol(x,y,center[ci,])

            # Calculate unit vectors e_r and e_phi
            e_r = np.array([np.cos(phi), np.sin(phi)])
            e_phi= np.array([-np.sin(phi), np.cos(phi)])

            # Calculate u and v 
            u[0,i] = ds.u_r[0,i] * e_r[0] + ds.u_phi[0,i] * e_phi[0]
            v[0,i] = ds.u_r[0,i] * e_r[1] + ds.u_phi[0,i] * e_phi[1]

    return u,v


def get_bg_wind(ds):
    """
    Calculate the background wind over the whole region. 
    Here the mean is taken over each level for u and v.

    Keyword arguments
    dataset -- Contains u and v as 3d variables of time, height and cells
    """

    # Extract u,v without time coordinate    
    da_u = ds.u[0]
    da_v = ds.v[0]
    # number of levels
    nlev = len(da_u.height)
    
    bg_u = np.empty([nlev])
    bg_v = np.empty([nlev])

    for i in range(0, nlev):
        bg_u[i] = da_u.values[i].mean()
        bg_v[i] = da_v.values[i].mean()
    
    bg_wind = [bg_u, bg_v]
    
    return bg_wind
