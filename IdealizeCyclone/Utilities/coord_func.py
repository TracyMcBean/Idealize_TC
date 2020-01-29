import numpy as np

def cart2pol(x,y,origin):
    '''
    Convert cartesian coordinates to polar coordinates.
    x,y are coordinates
    origin is centerline position on a single level
    '''
    x0=origin[0]
    y0=origin[1]
    rho = np.sqrt((x-x0)**2 + (y-y0)**2)
    phi = np.arctan2((y-y0), (x-x0))
    return(rho, phi)
    
