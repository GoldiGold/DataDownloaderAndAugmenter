import nibabel as nib
import numpy as np
import torch

from FinalConsts import NEW_FILES, FILES_KEYS

import os


def convert_scan_to_tensor(scan_path, tensor_path):
    scan = nib.load(scan_path).get_fdata()

    if len(scan.shape) == 3:
        scan = scan[..., np.newaxis]  # adding a channel dim to the last shape_dimension, like it is in a Nifti scan

    scan = torch.from_numpy(scan)

    #  moving the channels dimension to be the first one (index 0)
    scan = scan.permute(3, 0, 1, 2)  # dim3->dim0. dim0->dim1, dim1->dim2, dim2->dim3
    scan = scan.type(torch.float32)
    torch.save(scan, tensor_path)


def convert_all_scan_type_scans_to_tensors(scan_type: str, scans_path: str, tensors_path: str):
    scans_ids = os.listdir(scans_path)

    for scan_id in scans_ids:
        scan_path = os.path.join(scans_path, scan_id, NEW_FILES[scan_type])
        tensor_path = os.path.join(tensors_path, f'{scan_id}.pt')  # it's a pt file - tensor

        convert_scan_to_tensor(scan_path, tensor_path)


def convert_all_scans_to_tensors(scans_paths: dict, tensors_paths: dict):
    for scan_type in FILES_KEYS:
        if scan_type == 'rgb':
            continue  # they are not ready yet
        else:
            os.makedirs(tensors_paths[scan_type], exist_ok=True)
            convert_all_scan_type_scans_to_tensors(scan_type, scans_paths[scan_type], tensors_paths[scan_type])
