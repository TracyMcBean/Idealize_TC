import numpy as np
import math
import random

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

def blend_func(x, x_0, k=1, L=1):
    """ Logistic function to blend the idealized data with the original data.
    Calculated the probability of a given radius (x) to be accepted. Smaller radius will has higher probability of being accepted.

    Keyword
    float  -- variable for which probability of acceptenace should be calculated
    float  -- midpoint of curve
    int    -- Steepness of curve (default 1)
    int    -- curves maximum value (default 1)

    Returns probability between 0 and 1
    """
    
    p = L / (1 + math.exp(k * ( x - x_0) ))

    return p

def sel_blending(r, r_rad, val_ideal, val_orig, r_earth=6371):
    """
    Choose if original or idealized data should be saved for a given cell.
    
    Keyword arguments:
    float -- radius of cell
    float -- idealized value of cell
    float -- original value of cell
    """
    
    # Set midpoint for the logistic function
    km25_rad = 25/r_earth    
    x0 = r_rad - km25_rad
    # Get probability of acceptance
    p = blend_func(r, x0, k=1100)

    p_rand = random.uniform(0,1)
    
    # To check selection print which data was chosen
    verbose=False
    if p < 60 and verbose:
        print('R: %s and p: %s' % (r, p))
        if p_rand < p:
            print('ooo selected idealized data')
        else:
            print('--- selected original data')

    return val_ideal if p_rand < p else val_orig 
