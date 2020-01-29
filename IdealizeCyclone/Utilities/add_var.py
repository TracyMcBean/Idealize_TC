'''
Calculate variables to add to a given data set
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
    print('Calculate u_r and u_theta for following levels: %s to %s' % (ds.z_ifc.values[-len(center),0], ds.z_ifc.values[-1,0]))
    return ds
