import numpy as np
import nibabel as nib

import dipy.reconst.dti as dti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa

from ChangeScanSpacing import create_scan_with_new_spacing
import os

from FinalConsts import DIFFUSION_FILES, DIFFUSION_KEYS, DATASET_125, NEW_FILES


def create_diffusion_files_dict(diffusion_files_dir_path):
    diffusion_files_paths = {}

    for key in DIFFUSION_KEYS:
        diffusion_files_paths[key] = os.path.join(diffusion_files_dir_path, DIFFUSION_FILES[key])

    return diffusion_files_paths


def create_rgb_and_fa_image(diffusion_files_paths: dir, new_rgb_path: str, new_fa_path: str):
    # tic = time.time()
    '''
    The function gets the paths to all the files needed to create a single RGB file
    :param diffusion_files_paths: a dict with the keys: 'dwi', 'bvals', 'bvecs, 'fa', 'pdd', 'mask'
    :param new_rgb_path: the path of the new RGB file that will be created
    :param new_mask_path: the path of the mask that will fit the new RGB file that will be created (used in the change_scan_spacing)
    MAYBE WON'T BE NEEDED SINCE I WANT TO SEPERATE THE RGB CREATION AND THE SPACING MODIFICATION TO 2 DIFFERENT FUNCTIONS.
    :return: the RGB nifti image and the FA nifti image
    '''

    print('Load DWI data ... ', end='')
    dwi_scan = nib.load(diffusion_files_paths['dwi'])
    dwi_data = dwi_scan.get_fdata().astype(np.float32)

    # print('Load mask ... ', end='')
    mask_data = nib.load(diffusion_files_paths['mask']).get_fdata().astype(np.float32)

    dwi_masked_data = dwi_data * mask_data[..., np.newaxis]

    print('Load BVALs & BVECs data ... ', end='')
    bvals, bvecs = read_bvals_bvecs(diffusion_files_paths['bvals'], diffusion_files_paths['bvecs'])
    gradient_table_data = gradient_table(bvals, bvecs)

    print('Fitting DTI model ... ', end='')
    tensor_model = dti.TensorModel(gradient_table_data)
    tensor_fit = tensor_model.fit(dwi_masked_data)

    del dwi_masked_data

    print('Compute FA ... ', end='')
    FA = fractional_anisotropy(tensor_fit.evals).astype(np.float32)
    # print(FA.shape, FA.dtype)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    fa_img = nib.Nifti1Image(FA.astype(np.float32), dwi_scan.affine)
    nib.save(fa_img, new_fa_path)
    # for optimizing space: TODO: DELETE THIS COMMENT SO WE WILL SEE THE SPACE SAVING
    # del fa_img
    print('Compute RGB ... ', end='')
    RGB = color_fa(FA, tensor_fit.evecs)

    rgb_img = nib.Nifti1Image(RGB.astype(np.float32), dwi_scan.affine)
    nib.save(rgb_img, new_rgb_path)


def create_rgb_and_fa_files(diffusion_files_paths: dir, new_rgb_path: str, new_fa_path: str, new_mask_path: str,
                            new_spacing=1.25, tensor_polarity: bool = True):
    # creating files the RGB anf FA files straight from the diffusion files without modifications
    if not os.path.isfile(diffusion_files_paths['dwi']):
        print(f'couldnt create rgb for file {new_rgb_path}')
        return
    create_rgb_and_fa_image(diffusion_files_paths, new_rgb_path, new_fa_path)
    # modifying the newly created RGB&FA files to have the new spacing (that is why the old and new path are the same)
    create_scan_with_new_spacing(new_fa_path, new_fa_path, new_mask_path, new_spacing)
    create_scan_with_new_spacing(new_rgb_path, new_rgb_path, new_mask_path, new_spacing)


def create_all_rgb_and_fa_scans(diffusion_files_dir_path: str, new_scans_path: str, new_masks_path: str, new_spacing=1.25):
    diffusion_files_ids = sorted(os.listdir(diffusion_files_dir_path))
    # TODO: REMOVE THE SORTED AND THE 394, IT'S ONLY BECAUSE THE PROGRAM CRUSHED B4
    # TODO: MAYBE ADD SORTED TO EVERY os.listdir WE HAVE.
    for scan_id in diffusion_files_ids[394:]:
        id_diffusion_files_path = os.path.join(diffusion_files_dir_path, scan_id, 'Diffusion')
        diffusion_files_paths = create_diffusion_files_dict(id_diffusion_files_path)

        new_rgb_path = os.path.join(new_scans_path, scan_id, DIFFUSION_FILES['rgb'])
        new_fa_path = os.path.join(new_scans_path, scan_id, DIFFUSION_FILES['fa'])
        new_mask_path = os.path.join(new_masks_path, scan_id, NEW_FILES['brain_mask'])

        # create the dirs that will include the scan file
        os.makedirs(os.path.join(new_scans_path, scan_id), exist_ok=True)

        create_rgb_and_fa_files(diffusion_files_paths, new_rgb_path, new_fa_path, new_mask_path, new_spacing)

