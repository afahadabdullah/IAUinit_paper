import xarray as xr
import sys

# File path
file_path = '/home6/afahad/nobackup27/exp/IAU_exp/GEOSMIT_ME05062/holding/geosgcm_surf/200505/GEOSMIT_ME0506.geosgcm_surf.20050506_0130z.nc4'

try:
    ds = xr.open_dataset(file_path)
    print("Variables in file:")
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].attrs.get('long_name', 'No long_name')}")
    print("\nCoordinates:")
    print(ds.coords)
except Exception as e:
    print(f"Error reading file: {e}")
