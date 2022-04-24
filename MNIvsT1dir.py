import nibabel as nib
import os
import numpy as np
import consts
from nibabel.testing import data_path
from nilearn import plotting

sample_id = 100307
postfix = consts.NIFTI_POSTFIX
# MNI_dir = os.path.join(consts.DATASET_DIR, str(sample_id), 'MNINonLinear')
# T1_dir = os.path.join(consts.DATASET_DIR, str(sample_id), 'T1w')
segmap_subdir = consts.MASK_NAME
brain_subdir_T1w = 'T1w_acpc_dc_restore_brain.nii.gz'
brain_subdir_MNI = 'T1w_restore_brain.nii.gz'
mask_subdir = 'brainmask_fs.nii.gz'
brain_path = os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT['t1w'], '100307', 'T1w.nii.gz')


def main():
    # MNI_mask = os.path.join(MNI_dir, segmap_subdir)
    img = nib.load(brain_path)
    print(img.shape, img.get_data_dtype(), img.affine.shape,
          f'img 3D size is: 1st X 2nd X 3rd = {img.shape[0] * img.shape[1] * img.shape[2]}')
    print(f'sizes: ({img.shape[0]},{img.shape[1]},{img.shape[2]})')
    hdr = img.header
    print(hdr.get_xyzt_units(), type(hdr), '\n', hdr.get_zooms())

    '''
    view2 = plotting.plot_anat(img)
    print(type(view2))
    # view2.open_in_browser()
    # view22 = plotting.view_img(view2)

    view = plotting.view_img(img)
    view.open_in_browser()
    # THIS IS TO SHOW A SCAN IN THE BROWSER
    # view22.open_in_browser()
    '''
    vals = img.get_fdata()
    print(type(vals), vals.shape, np.max(vals), np.min(vals))
    # segmap_copy = np.copy(vals)

    return 'smile'


if __name__ == '__main__':
    main()
