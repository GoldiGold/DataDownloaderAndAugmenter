import os

import nibabel as nib

from FinalConsts import DATASET_125, NEW_FILES, FILES_KEYS, T1W_MAX, T1W_MIN
from ChangeScanSpacing import SCANS_DTYPE


def normalize_t1w_scan(t1w_scan_path):
    t1w_scan = nib.load(t1w_scan_path)
    t1w_data = t1w_scan.get_fdata()

    t1w_data = (t1w_data - T1W_MIN) / (T1W_MAX - T1W_MIN)
    nib.save(nib.Nifti1Image(t1w_data.astype(SCANS_DTYPE), t1w_scan.affine), t1w_scan_path)


def normalize_t1w_scans(t1w_scans_path):
    t1w_ids = os.listdir(t1w_scans_path)

    for t1w_id in t1w_ids:
        t1w_scan_path = os.path.join(t1w_scans_path, t1w_id, NEW_FILES['t1w'])
        normalize_t1w_scan(t1w_scan_path)

# DON'T NEED IT NOW BECAUSE THE VALUES ARE ALREADY BETWEEN 0-1
# def normalize_rgb_scans(rgb_scans):
