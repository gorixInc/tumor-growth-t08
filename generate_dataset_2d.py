# %%
from glob import glob
import pydicom as dicom
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
# %%
data_folder = Path('raw_data/PDMR-Texture-Analysis')
# %%
level1_paths = glob(str(data_folder/'*'))
data_dict = {}
for lvl1_path in level1_paths:
    level2_paths = glob(f'{lvl1_path}/*')
    lvl2_dict = {}
    for lvl2_path in level2_paths:
        lvl3_dict = {}
        lvl3_paths = glob(f'{lvl2_path}/*')
        for scan_path in lvl3_paths:
            dcm_slices = glob(f'{scan_path}/*.dcm')
            lvl3_dict[Path(scan_path).name] = dcm_slices
        lvl2_dict[Path(lvl2_path).name] = lvl3_dict
    data_dict[Path(lvl1_path).name] = lvl2_dict
# %%
dataset_meta = {
    'Xy_path': [],
}

def process_scan(scan_image_paths, img_id, metadata, min_frac_avg = 0.5):
    scan_path = Path(scan_image_paths[0]).parents[0]
    segmentation_paths = glob(str(scan_path) + f'/Segmentation*.nii')
    if not len(segmentation_paths) == 1:
        return img_id
    segmentation_img = nib.load(segmentation_paths[0]).get_fdata()
    if not segmentation_img.shape[2] == len(scan_image_paths):
        return img_id
    # Flitering out some segmentation files that contain full scans for some reason
    if np.max(segmentation_img > 1):
        return img_id
    average_area = np.mean(np.sum(segmentation_img, axis=(0, 1)))
    lowest_bound = average_area*min_frac_avg
    for i in range(len(scan_image_paths)):
        ds = dicom.dcmread(scan_image_paths[i])
        slice_array = np.array(ds.pixel_array)
        ann_array = np.array(segmentation_img[:, :, i]).T
        if np.sum(ann_array) < lowest_bound:
            continue
        if not np.any(ann_array):
            continue
        xy = np.array([slice_array, ann_array])
        xy_path = f'2d_dataset/data/Xy_{img_id}.npy'
        np.save(xy_path, xy)
        metadata['Xy_path'].append(xy_path)
        img_id += 1
    return img_id

img_id = 0
lvl1_keys = list(data_dict.keys())
for key1 in lvl1_keys:
    lvl2_data = data_dict[key1]
    lvl2_keys = list(lvl2_data.keys())
    for key2 in lvl2_keys:
        lvl3_data = data_dict[key1][key2]
        lvl3_keys = list(lvl3_data.keys())
        for scan_name in lvl3_keys:
            scan_image_paths = data_dict[key1][key2][scan_name]
            if len(scan_image_paths) < 2:
                continue 
            img_id = process_scan(scan_image_paths, img_id, dataset_meta)
df = pd.DataFrame(dataset_meta)
df.to_csv('2d_dataset/dataset_meta.csv', sep=',')
# %%
def load_data_2d(metadata_df):
    X, y = [], []
    for i in range(len(metadata_df)): 
        Xy_path = metadata_df['Xy_path'].iloc[i]
        Xy = np.load(Xy_path)
        X.append(Xy[0])
        y.append(Xy[1])
    return X, y
# %%
meta_df = pd.read_csv('2d_dataset/dataset_meta.csv')
X, y = load_data_2d(meta_df.iloc[np.random.randint(0, len(meta_df), 10)])
for i in range(10):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(X[i])
    ax[1].imshow(y[i])

# %%
