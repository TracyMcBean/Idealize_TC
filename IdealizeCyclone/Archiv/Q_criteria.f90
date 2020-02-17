program q_criteria
  use netcdf
  implicit none
  
  character (len = *), parameter :: FILE_NAME = &
    "/scratch/usr/bekthkis/ICON_08_2019/Fiona2016/init_data/dei4_NARVALII_2016081700_fg_DOM01_ML_0012.nc"
  integer :: ncid
  integer, parameter :: ncells = 2469998 
  integer, parameter :: nlev = 75
  integer, parameter :: ntime = 1 
  integer :: u_id, v_id, lat_id, lon_id
  ! Arrays for data
  real :: u_in(ncells, nlev, ntime), v_in(ncells,nlev, ntime) 
  real :: lat_in(ncells, nlev), lon_in(ncells, nlev) 

  integer, dimension(1) :: lat_max_id, lon_min_id
  real    :: lat_max_val, lon_min_val
 
  ! Open file
  call check( nf90_open(FILE_NAME, nf90_nowrite, ncid))

  ! Get Wind Components
  call check( nf90_inq_varid(ncid, "u", u_id))
  call check( nf90_inq_varid(ncid, "v", v_id))

  call check( nf90_get_var(ncid, u_id, u_in))
  call check( nf90_get_var(ncid, v_id, v_in))

  ! Get lon lat coordinates
  call check( nf90_inq_varid(ncid, "clon", lon_id))
  call check( nf90_inq_varid(ncid, "clat", lat_id))

  call check( nf90_get_var(ncid, lon_id, lon_in))
  call check( nf90_get_var(ncid, lat_id, lat_in))

  print *, shape(lon_in(:,1))
  print *, lon_in(1900:2000,1)
  print *, lat_in(1000:1500,1)
 
  lat_max_id = maxloc(lat_in(:,1))
  lat_max_val = maxval(lat_in)
  lon_min_id = minloc(lon_in(:,1))
  lon_min_val = minval(lon_in)

  print *, lat_max_id, lat_max_val, lon_min_id, lon_min_val

contains
  subroutine check(status)
    integer, intent ( in) :: status
    
    if(status /= nf90_noerr) then 
      print *, trim(nf90_strerror(status))
      stop "Stopped"
    end if
  end subroutine check    
end program q_criteria
