import os

from ChangeScanSpacing import *
from NormalizeScans import *
from FinalConsts import DATASET_07, DATASET_125, NEW_FILES, SCANS_KEYS


# def create_a_single_scan()

def create_all_masks(old_masks_path, new_masks_path, new_spacing=1.25):
    masks_ids = os.listdir(old_masks_path)

    for mask_id in masks_ids:
        old_mask_path = os.path.join(old_masks_path, mask_id, NEW_FILES['barin_masks'])
        new_mask_path = os.path.join(new_masks_path, mask_id, NEW_FILES['barin_masks'])
        # create the dirs that will include the mask file
        os.makedirs(new_mask_path, exist_ok=True)

        create_mask_with_new_spacing(old_mask_path, new_mask_path, new_spacing)


def create_all_scans_of_single_type(old_scans_path, new_scans_path, new_masks_path, scan_type: str, new_spacing=1.25):
    old_scans_ids = os.listdir(old_scans_path)

    for scan_id in old_scans_ids:
        old_scan_path = os.path.join(old_scans_path, scan_id, NEW_FILES[scan_type])
        new_scan_path = os.path.join(new_scans_path, scan_id, NEW_FILES[scan_type])
        new_mask_path = os.path.join(new_masks_path, scan_id, NEW_FILES['barin_masks'])

        # create the dirs that will include the mask file
        os.makedirs(new_scan_path, exist_ok=True)

        create_scan_with_new_spacing(old_scan_path, new_scan_path, new_mask_path, new_spacing)


def create_all_scans_for_dataset(old_scans_paths: dict, new_scans_paths: dict, new_mask_path, new_spacing=1.25):
    for scan_type in SCANS_KEYS:
        create_all_scans_of_single_type(old_scans_paths[scan_type], new_scans_paths[scan_type], new_mask_path,
                                        scan_type, new_spacing)

        if scan_type == 't1w':
            normalize_t1w_scans(new_scans_paths[scan_type])


def create_dataset(old_dataset_paths: dict, new_dataset_paths: dict, new_spacing=1.25):
    if 'brain_masks' not in old_dataset_paths.keys() or 'brain_masks' not in new_dataset_paths.keys():
        print(f'COULDN\'T CREATE NEW DATASET BECAUSE NO MASKS PATHS WERE GIVEN')
        return 0

    create_all_masks(old_dataset_paths['brain_masks'], new_dataset_paths['brain_masks'], new_spacing)
    create_all_scans_for_dataset(old_dataset_paths, new_dataset_paths, new_dataset_paths['brain_masks'], new_spacing)
