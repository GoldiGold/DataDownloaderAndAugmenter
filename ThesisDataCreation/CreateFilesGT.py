import nibabel as nib
import numpy as np

import os

from FinalConsts import NEW_FILES, VALUE_TO_CHANNEL_MAPPING, VALUE_TO_CHANNEL_MAPPING_WM


def create_gt_scan(old_gt_path: str, new_gt_path: str, value_to_channel_mapping: dict):
    '''
    Gets an aparc+aseg nifti scan (a 3d scan with voxels values representing the areas in the brain)
    :param old_gt_path: the path to the old ground truth scan
    :param new_gt_path: the path to the new ground truth scan
    :param value_to_channel_mapping: a dictionary that maps a voxel value to a channel in the new GT scan.
    :return:
    '''

    old_gt_img = nib.load(old_gt_path)
    old_gt_data = old_gt_img.get_fdata()

    # amount of unique channel values in the mapping + Background channel
    amount_of_channels = len(set(value_to_channel_mapping.values())) + 1
    # *old_gt_data.shape takes the values themselves from the shape tuple
    new_gt_data = np.zeros(shape=(*old_gt_data.shape, amount_of_channels), dtype=np.float32)

    for x_voxel in range(old_gt_data.shape[0]):
        for y_voxel in range(old_gt_data.shape[1]):
            for z_voxel in range(old_gt_data.shape[2]):
                voxel_value = old_gt_data[x_voxel, y_voxel, z_voxel]
                if voxel_value in value_to_channel_mapping.keys():
                    # if part of the values we want to create a new channel for them (to segment)
                    new_gt_data[x_voxel, y_voxel, z_voxel, value_to_channel_mapping[voxel_value]] = 1
                elif voxel_value != 0:
                    # if part of the brain but not of the segmented values - part of the BG channel inside the brain.
                    new_gt_data[x_voxel, y_voxel, z_voxel, 0] = 1

    nib.save(nib.Nifti1Image(new_gt_data, old_gt_img.affine), new_gt_path)


def create_all_gt_scans(old_scans_path: str, new_scans_path: str):
    gt_ids = os.listdir(old_scans_path)

    for scan_id in gt_ids:
        old_gt_path = os.path.join(old_scans_path, scan_id, NEW_FILES['general_mask'])
        new_gt_path = os.path.join(new_scans_path, scan_id, NEW_FILES['gt'])

        os.makedirs(os.path.join(new_scans_path, scan_id), exist_ok=True)

        create_gt_scan(old_gt_path, new_gt_path, VALUE_TO_CHANNEL_MAPPING_WM)
        print(f'created {scan_id} gt')
