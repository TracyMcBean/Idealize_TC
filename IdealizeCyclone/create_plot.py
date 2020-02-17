''' Create plots of whatever you want to see Yippieeyay :D
'''
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# initial data
ds = xr.open_dataset("/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc")

# Centerline ------------------------------------------------------------------
plot_centerline = True

center = np.load('./Data/center_fiona.npy')
ds = xr.open_dataset("/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc")

print(ds.z_ifc.values[(34+9):75,0], 'length: ', len(ds.z_ifc.values[(34+9):75,0]))

# Plot centerline
if plot_centerline:                                 
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.plot(np.flipud(center[:,0]), np.flipud(center[:,1]), np.flipud(ds.z_ifc.values[34:75,0]))
     # Set number of ticks for lon/lat axes
     plt.locator_params(axis='y', nbins=5)
     plt.locator_params(axis='x', nbins=5)       
     plt.title('Centerline of Fiona (17.08.2016, 12 UTC)')
     ax.set_xlabel('Longitude [rad]')
     ax.set_ylabel('Latitude [rad]')
     ax.set_zlabel('Height at half level [m]')
     #plt.show()

     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.plot(np.flipud(center[9:41,0]), np.flipud(center[9:41,1]), np.flipud(ds.z_ifc.values[(34+9):75,0]))
     # Set number of ticks for lon/lat axes
     plt.locator_params(axis='y', nbins=4)
     plt.locator_params(axis='x', nbins=4)       
     plt.title('Centerline of Fiona (17.08.2016, 12 UTC)')
     ax.set_xlabel('Longitude [rad]')
     ax.set_ylabel('Latitude [rad]')
     ax.set_zlabel('Height at half level [m]')
     #plt.show()


     fig = plt.figure()
     ax = fig.add_subplot(121)
     ax.plot(np.flipud(center[9:41,0]), np.flipud(ds.z_ifc.values[(34+9):75,0]))
     # Set number of ticks for lon/lat axes
     plt.locator_params(axis='x', nbins=4)
     #plt.title('Longitude of centerline (17.08.2016, 00 UTC)')
     ax.set_xlabel('Longitude [rad]')
     ax.set_ylabel('Height at half level [m]')
     
     ax = fig.add_subplot(122)
     ax.plot(np.flipud(center[9:41,1]), np.flipud(ds.z_ifc.values[(34+9):75,0]))
     # Set number of ticks for lon/lat axes
     plt.locator_params(axis='x', nbins=4)
     #plt.title('Latitude of centerline (17.08.2016, 12 UTC)')
     ax.set_xlabel('Latitude [rad]')

     plt.suptitle('Centerline of Fiona (17.08.2016, 12 UTC)')
     plt.show()

