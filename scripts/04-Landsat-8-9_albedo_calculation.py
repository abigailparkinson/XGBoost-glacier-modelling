import rasterio
import numpy as np
import os
from glob import glob
from collections import defaultdict

# define input directory containing Landsat reflectance bands
# and output directory for albedo rasters
root = r'C:\Users\p09367ap\OneDrive - The University of Manchester\ERP\classification'
out_dir = os.path.join(root, 'albedo_outputs')
os.makedirs(out_dir, exist_ok=True)

# group landsat bands by acquisition date using the file naming convention
groups = defaultdict(dict)

for f in glob(os.path.join(root, '*.tif')):
    base = os.path.basename(f)
    parts = base.split('_')
    date = f'{parts[0]}_{parts[1]}'
    band = parts[2].split('.')[0]
    groups[date][band] = f

# function to load individual landsat bands and raster metadata
# ensures the calculation can be used for any imagery from any time period
def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype('float32'), src.profile

# process each landsat acquisition separately
for date, band_dict in sorted(groups.items()):
    print(f'Processing {date}')
    required = ['b2', 'b4', 'b5', 'b6', 'b7']
    missing = [b for b in required if b not in band_dict]
    if missing:
        print(f'Skipping {date} - missing bands: {missing}')
        continue

# load the necessary bands for albedo calculation
    b2, profile = load_band(band_dict['b2'])
    b4, _ = load_band(band_dict['b4'])
    b5, _ = load_band(band_dict['b5'])
    b6, _ = load_band(band_dict['b6'])
    b7, _ = load_band(band_dict['b7'])

# convert Landsat Collection 2 digital numbers to surface reflectance
    scale = 0.0000275
    offset = -0.2
    b2 = b2*scale + offset
    b4 = b4*scale + offset
    b5 = b5*scale + offset
    b6 = b6*scale + offset
    b7 = b7*scale + offset

# equation for albedo derivation from Liang et al., 2001
    albedo = (
        0.356*b2 + 0.13*b4 + 0.373*b5 + 0.085*b6 + 0.072*b7 - 0.0018
    )
# restricting albedo to the physical range
    albedo = np.clip(albedo, 0, 1)

# configure GeoTIFF export settings
    profile.update(dtype='float32', count = 1)
    out_file = os.path.join(out_dir, f'albedo_{date}.tif')

# export the broadband albedo raster for the relevant acquisition date
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(albedo.astype('float32'), 1)
    print(f'Saved {out_file}')

print('All done!')
