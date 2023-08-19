# built in
import os
from os.path import join as opj
import multiprocessing

# 3rd party
import numpy as np
import nibabel as nib

# dipy
import dipy.reconst.dti as dti
from dipy.io import read_bvals_bvecs
from dipy.align.imaffine import AffineMap
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa

# misc
import constsDeprecated
import FinalConsts
from FinalConsts import NEW_FILES, FILES_KEYS, DATASET_07, DATASET_125
from constsDeprecated import rgb_brains
import time

MASK_SIZES = {'HCP': (260, 311, 260), 'HARDI': (145, 174, 145), 'HARDI_NOA': (144, 160, 144), 'CLINICAL': (73, 87, 73)}


def update_affine_matrix(img_in: nib.Nifti1Image, old_shape, new_spacing=1.25):
    # old_shape = img_data.shape
    img_spacing = abs(img_in.affine[0, 0])

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)
    new_affine[0, 0] = new_spacing if img_in.affine[0, 0] > 0 else -new_spacing
    new_affine[1, 1] = new_spacing if img_in.affine[1, 1] > 0 else -new_spacing
    new_affine[2, 2] = new_spacing if img_in.affine[2, 2] > 0 else -new_spacing

    if new_spacing == 1.25:
        new_shape = MASK_SIZES['HARDI_NOA']
    elif new_spacing == 1.5:
        new_shape = MASK_SIZES['HARDI_NOA']
    elif new_spacing == 2.5:
        new_shape = MASK_SIZES['CLINICAL']
    else:
        # original version
        new_shape = np.floor(np.array(img_in.get_fdata().shape) * (img_spacing / new_spacing))
    new_shape = new_shape[:3]  # drop last dim

    affine_map = AffineMap(np.eye(4),
                           new_shape, new_affine,
                           old_shape, img_in.affine)

    return affine_map, new_affine


def change_spacing_4d(img_in: nib.Nifti1Image, new_spacing=1.25):
    data = img_in.get_fdata()
    old_shape = data.shape

    image_3d = False
    if data.ndim == 3:
        image_3d = True
        data = data[..., np.newaxis]

    affine_map, new_affine = update_affine_matrix(img_in, old_shape, new_spacing)

    new_data = []
    for i in range(data.shape[3]):
        # Generally nearest gets better results than linear interpolation
        res = affine_map.transform(data[:, :, :, i], interpolation="nearest")
        new_data.append(res)

    new_data = np.array(new_data).transpose(1, 2, 3, 0)

    if image_3d:
        new_data = new_data[..., 0]

    img_new = nib.Nifti1Image(new_data.astype(data.dtype), new_affine)

    return img_new


def create_mask_with_new_spacing(mask_to_modify_path: str, new_mask_path: str, new_spacing=1.25):
    '''
    Creates a new brain_mask file with the new_spacing specified as its spacing.
    This needs to run before converting other scans because they need a converted brain mask in order to work.
    :param mask_to_modify_path: the path to the brain mask it modifies
    :param new_mask_path: the path that the new scan will be saved to
    :param new_spacing: the new spacing of the scan
    :return:
    '''
    mask_to_modify = nib.load(mask_to_modify_path)
    new_mask_image = change_spacing_4d(mask_to_modify, new_spacing)
    nib.save(new_mask_image, new_mask_path)


def create_scan_with_new_spacing(scan_to_modify_path: str, modified_mask_path: str, new_scan_path: str,
                                 new_spacing=1.25):
    '''
    Gets a path to a nifti scan (nii.gz) and modify its spacing to the new spacing it got.
    :param scan_to_modify_path: the path to the scan it modifies
    :param new_scan_path: the path that the new scan will be saved to
    :param modified_mask_path: the path to the mask we use after thr spacing change in order to clean the scan
    :param new_spacing: the new spacing of the scan
    :return: at this point it doesn't return anything, in the future it may throw exceptions when failing.
    '''

    scan_to_modify = nib.load(scan_to_modify_path)
    new_scan_nifti_image = change_spacing_4d(scan_to_modify, new_spacing)
    modified_mask = nib.load(modified_mask_path)

    new_scan_data = new_scan_nifti_image.get_fdata()
    modified_mask_data = modified_mask.get_fdata()
    modified_mask_data[modified_mask_data < 0] = 0

    if len(new_scan_data.shape) == 4:
        for dim in range(new_scan_data.shape[-1]):
            new_scan_data[:, :, :, dim] = new_scan_data[:, :, :, dim] * modified_mask_data

    elif len(new_scan_data.shape) == 3:
        new_scan_data = new_scan_data * modified_mask_data

    else:
        raise ValueError(f'shape isn\'t 3 r 4 but: {new_scan_data.shape}')

    new_scan_image = nib.Nifti1Image(new_scan_data.astype(np.float32), new_scan_nifti_image.affine)
    nib.save(new_scan_image, new_scan_path)


#  TODO: NEED TO CREATE A FUNCTION THAT CREATES ALL THE SCANS OF A SINGLE BRAIN
#  TODO: NEED TO CREATE A FUNCTION THAT CREATES ALL THE SCANS OF ALL THE BRAINS

def create_mask_from_t1w(t1w_path: str, new_mask_path: str):
    t1w_scan = nib.load(t1w_path)
    data = t1w_scan.get_fdata()
    mask = data.copy()
    mask[data != 0] = 1
    mask_image = nib.Nifti1Image(mask, t1w_scan.affine)
    nib.save(mask_image, new_mask_path)


if __name__ == '__main__':
    file_key = 't1w'
    # THIS WAS A TEST TO SEE IF IT WORKS, USING A FILE I ALREADY CONVERTED IN THE PAST (PATIENT 100206)
    # create_mask_with_new_spacing(opj(DATASET_07['brain_masks'], '100206', NEW_FILES['brain_masks']), 'temp.nii.gz',
    #                              1.25)
    # create_scan_with_new_spacing(opj(DATASET_07['t1w'], '100206', NEW_FILES['t1w']), modified_mask_path='temp.nii.gz',
    #                              new_scan_path='scan_temp.nii.gz', new_spacing=1.25)

    hadassa_t1w_path = '/home/chen/Downloads/HadasaBrainData/T1_MNI_125mm_extracted.nii.gz'
    hadassa_mask_path = '/home/chen/Downloads/HadasaBrainData/Hadassa_mask.nii.gz'
    create_mask_from_t1w('/home/chen/Downloads/HadasaBrainData/T1_MNI_125mm_extracted.nii.gz', '/home/chen/Downloads/HadasaBrainData/Hadassa_mask.nii.gz')

    modified_mask_path = '/home/chen/Downloads/HadasaBrainData/Hadassa_mask_in_shape.nii.gz'
    modified_t1w_path = '/home/chen/Downloads/HadasaBrainData/Hadassa_t1w_in_shape.nii.gz'
    create_mask_with_new_spacing(hadassa_mask_path, modified_mask_path)
    create_scan_with_new_spacing(hadassa_t1w_path, modified_mask_path, modified_t1w_path)