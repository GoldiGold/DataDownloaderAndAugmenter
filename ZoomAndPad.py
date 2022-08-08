# 3rd party
import numpy as np
import nibabel as nib

# misc
import consts
import os


def check_zoom_on_all(t1w_dir: str, brain_file):
    faulty_counter = 0
    brains = sorted([str(i) for i in os.listdir(t1w_dir) if i.isdigit()])
    for brain in brains:
        scan = nib.load(os.path.join(t1w_dir, brain, brain_file)).get_fdata()
        trim_scan = trim_zeros(scan)
        required_shape = (128, 144, 128)
        faulty_idx = [int(trim_scan.shape[i] > required_shape[i]) for i in range(len(required_shape))]
        if sum(faulty_idx) > 0:
            print(f'ERROR: brains {brain} size is too big: {trim_scan.shape}, fault: {faulty_idx}')
            faulty_counter += 1
    print(f'Faulty brain scans amount is: {faulty_counter}')


def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]


def zoom(arr: np.ndarray):
    '''
    This function zooms into the subject in the array meaning:
    it removes any full rows of zeros from each dimension until the "subject" in the "center" of the array
    reaches the edges
    :param arr: an array of some dimension
    :return: the array "clipped"/"zoomed" into the non zero values
    '''
    positives = np.where(arr > 0, 1, 0)

    zero_across_rows = np.argwhere(np.all(arr == 0, axis=(1, 2)))
    # arr = np.delete(arr, zero_across_rows, axis=(1, 2))
    zero_across_cols = np.argwhere(np.all(arr == 0, axis=(0, 2)))
    zero_across_depth = np.argwhere(np.all(arr == 0, axis=(0, 1)))
    # print(arr[i, ..., :].shape, '\n', arr[i, ..., :])
    # arr_dup = np.
    print(zero_across_rows, zero_across_cols, zero_across_depth, sep='\n')
    # print(zero_indices[0][:, 0], '\n', zero_indices[0][:, 0], '\n', zero_indices[0].shape, len(zero_indices))
    return arr


if __name__ == '__main__':
    # check_zoom_on_all(os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT_125['t1w']))
    check_zoom_on_all(os.path.join(consts.SSD_DATASET, 'T1w-1.25'), brain_file='T1w.nii.gz')
    # t1 = nib.load('/home/cheng/Desktop/Dataset/T1w-1.25/992774/T1w.nii.gz')
    # t1_data = t1.get_fdata()
    # t1_noa = nib.load('/home/cheng/Desktop/Dataset/T1-NOA-1.25/992774/T1-NOA.nii.gz')
    # t1_noa_data = t1_noa.get_fdata()
    # brain_mask_noa = nib.load('/home/cheng/Desktop/Dataset/All-Brain-Masks-1.25/992774/brain_mask.nii.gz')
    # brain_mask_noa_data = brain_mask_noa.get_fdata()
    # brain_mask = nib.load('/home/cheng/Desktop/Dataset/Brain-Masks-AGY-1.5/992774/brain_mask.nii.gz')
    # brain_mask_data = brain_mask.get_fdata()
    # # zoomed_mask = zoom(brain_mask)
    # trimmed_mask_data = trim_zeros(brain_mask_data)
    # trimmed_mask_noa_data = trim_zeros(brain_mask_noa_data)
    # print(trimmed_mask_data.shape, trimmed_mask_noa_data.shape)
    # trimmed_mask = nib.Nifti1Image(trimmed_mask_data, brain_mask.affine)
    # nib.save(trimmed_mask, 'TrimmedMask.nii.gz')
    # trimmed_mask_noa = nib.Nifti1Image(trimmed_mask_noa_data, brain_mask.affine)
    # nib.save(trimmed_mask_noa, 'TrimmedMaskNoa.nii.gz')
