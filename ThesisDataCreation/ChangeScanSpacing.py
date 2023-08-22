import os

import nibabel as nib
import numpy as np

from dipy.align.imaffine import AffineMap

mask_sizes = {'HCP': (260, 311, 260), 'HARDI': (145, 174, 145), 'HARDI_GOLD': (144, 160, 144), 'CLINICAL': (73, 87, 73),
              0.7: (260, 311, 260), 1.5: (144, 160, 144), 1.25: (144, 160, 144), 2.5: (73, 87, 73)}

# TODO: move to a consts file
SCANS_DTYPE = np.float32


def update_affine_matrix(img_in: nib.Nifti1Image, old_shape, new_spacing=1.25, tensor_polarity: bool = True):
    '''

    tensor_polarity: meaning we need to fit the values in the affine-matrix to values that would make the
    spatial-indexing same as the ones in the pytorch and tensorflow tensors
    (0,0,0 is the same as 0,0,0 in the mricron visual)
    if True the Diagonal is positive, and the last column is negative
    '''
    # old_shape = img_data.shape
    img_spacing = abs(img_in.affine[0, 0])

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)

    if tensor_polarity:
        # setting the spacing in the new affine to the positive value of the new_spacing
        new_affine[0, 0] = new_affine[1, 1] = new_affine[2, 2] = abs(new_spacing)
        # setting the scan_size in the new affine to the negative value of it so the tensor_polarity would fit
        new_affine[0, -1] = -abs(new_affine[0, -1])
        new_affine[1, -1] = -abs(new_affine[1, -1])
        new_affine[2, -1] = -abs(new_affine[2, -1])

    else:
        new_affine[0, 0] = new_spacing if img_in.affine[0, 0] > 0 else -new_spacing
        new_affine[1, 1] = new_spacing if img_in.affine[1, 1] > 0 else -new_spacing
        new_affine[2, 2] = new_spacing if img_in.affine[2, 2] > 0 else -new_spacing

    if new_spacing in mask_sizes.keys():
        new_shape = mask_sizes[new_spacing]
    else:
        # original version
        new_shape = np.floor(np.array(img_in.get_fdata().shape) * (img_spacing / new_spacing))
    new_shape = new_shape[:3]  # drop 4th and next dims (if exist)

    affine_map = AffineMap(np.eye(4),
                           new_shape, new_affine,
                           old_shape, img_in.affine)

    return affine_map, new_shape, new_affine


def change_scan_spacing(img_in: nib.Nifti1Image, new_spacing=1.25, tensor_polarity: bool = True):
    data = img_in.get_fdata()
    old_shape = data.shape

    is_image_3d = False
    if data.ndim == 3:
        is_image_3d = True
        data = data[..., np.newaxis]

    affine_map, new_shape, new_affine = update_affine_matrix(img_in, old_shape, new_spacing, tensor_polarity)

    new_data = []
    for i in range(data.shape[3]):
        # calculating the new values for the scan's new shape
        # Generally nearest a bit better results than linear interpolation
        res = affine_map.transform(data[:, :, :, i], interpolation="nearest")
        new_data.append(res)

    new_data = np.array(new_data).transpose((1, 2, 3, 0))

    if is_image_3d:
        new_data = new_data[..., 0]

    img_new = nib.Nifti1Image(new_data.astype(SCANS_DTYPE), new_affine)

    return img_new


def create_scan_with_new_spacing(old_scan_path: str, new_scan_path: str, new_mask_path: str, new_spacing=1.25,
                                 should_recreate: bool = False, tensor_polarity: bool = True):
    '''
    Receiving a 3d or 4d scan
    '''
    if not os.path.isfile(new_mask_path):
        print(f'THIS MASK FILE DOESN\'T EXIST {new_mask_path}')
        return 0

    if not os.path.isfile(old_scan_path):
        print(f'THIS SCAN FILE DOESN\'T EXIST {new_mask_path}')
        return 0

    old_scan = nib.load(old_scan_path)
    new_mask_data = (nib.load(new_mask_path)).get_fdata()

    new_scan = change_scan_spacing(old_scan, new_spacing, tensor_polarity)
    new_scan_data = new_scan.get_fdata()
    # removing the below zero values from the new scan
    new_scan_data[new_scan_data < 0] = 0

    is_scan_3d = False
    if len(new_scan.shape) == 3:
        # adding a channels dim for the scan
        is_scan_3d = True
        new_scan_data = new_scan_data[..., np.newaxis]

    for dim in range(new_scan_data.shape[-1]):  # -1 because in the nifti the channels dim is the last
        # removing the data that is out of the mask boundaries
        new_scan_data[:, :, :, dim] = new_scan_data[:, :, :, dim] * new_mask_data

    if is_scan_3d:
        # the scan was 3d and we return it to normal
        new_scan_data = new_scan_data[:3]

    nib.save(
        nib.Nifti1Image(new_scan_data.astype(SCANS_DTYPE), new_scan.affine),
        new_scan_path
    )

    print(f'Created file: {new_scan_path}')


def create_mask_with_new_spacing(mask_path: str, out_mask_path: str, new_spacing=1.25, tensor_polarity: bool = True):
    if not os.path.isfile(mask_path):
        print(f'THIS MASK FILE DOESN\'T EXIST {mask_path}')
        return 0

    old_mask = nib.load(mask_path)
    new_mask_image = change_scan_spacing(old_mask)
    nib.save(new_mask_image, out_mask_path)
