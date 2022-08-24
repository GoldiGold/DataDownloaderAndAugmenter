import nibabel as nib
import os
import consts
import T1wConsts
import numpy as np
import json
from createTables import get_dataset_names
# from nilearn import plotting

'''
In editing the brains mask we want to keep only the values of the LABELS in the consts files (and maybe not 
differentiate between left and right hemisphere)
'''


def update_single(lookup_table: dict, label2new_label: dict, label2value: dict, mask_path: str, bg_value: int = 0):
    '''
    This function takes the aparc+aseg mask and extract only the labels we want to our research, it also gives them new
    values according to the conversion table (that is created from 2 lookup tables (old - values in original and new -
    values we set to be in the new). and returns a new mask with only the values we want.
    :param mask_path:
    :param label2value:
    :param label2new_label:
    :param brain_id: the brain (patient) id number to find in the dataset
    :param lookup_table: the table that maps the original labels and the original values in the aprac+aseg file.
    :param conversion_table: a table between the values of the old -> new tables. state the new voxel values
    :param bg_value: a value for voxels inside the brain we are not interested in.
    :param data_dir: that path to the Dataset.
    :return: a new Nifti1Image OR WE WILL SAVE IT HERE ALREADY AND WE WON'T RETURN ANYTHING.
    '''
    #  mask_path = os.path.join(data_dir, str(brain_id), 'MNINonLinear', consts.MASK_NAME)
    mask = nib.load(mask_path)
    # print(mask.shape, mask.get_data_dtype(), mask.affine.shape,
    #       f'mask 3D size is: 1st X 2nd X 3rd = {mask.shape[0] * mask.shape[1] * mask.shape[2]}')
    hdr = mask.header
    # print(mask.file_map, hdr.get_xyzt_units(), type(hdr), '\n', hdr.get_zooms(), '\nheader:', hdr)
    value2label_orig = {value: key for key, value in lookup_table.items()}
    voxels = mask.get_fdata()
    # print(type(voxels), voxels.shape, np.max(voxels), np.min(voxels))
    vox_copy = np.copy(voxels)
    lookup_table_values = lookup_table.values()
    for i in range(vox_copy.shape[0]):
        for j in range(vox_copy.shape[1]):
            for k in range(vox_copy.shape[2]):
                # if i == 26 and j == 125 and k == 117:
                #     print(f'reached the bug')
                if voxels[i, j, k] not in lookup_table_values:
                    vox_copy[i, j, k] = 0 if voxels[i, j, k] == 0 else label2value[label2new_label['BACKGROUND']]
                else:
                    vox_copy[i, j, k] = label2value[  # The new value for the new label
                        label2new_label[  # The new label for the original label
                            value2label_orig[voxels[i, j, k]]]]  # The label of the original value
    print(f'for debug the max value is: {vox_copy.max()}')
    new_mask = nib.nifti1.Nifti1Image(vox_copy, mask.affine, header=hdr.copy())
    return new_mask

def update_single_multi_channel(lookup_table: dict, label2new_label: dict, label2value: dict, mask_path: str, bg_value: int = 0):
    '''
    This function takes the aparc+aseg mask and extract only the labels we want to our research, it also gives them new
    values according to the conversion table (that is created from 2 lookup tables (old - values in original and new -
    values we set to be in the new). and returns a new mask with only the values we want.

    THE NEW VALUES HERE HAVE TO BE IN ORDER (0,1,2) (3,4,5) SO WE CAN MAP THEM TO THE CHANNELS

    :param mask_path:
    :param label2value:
    :param label2new_label:
    :param brain_id: the brain (patient) id number to find in the dataset
    :param lookup_table: the table that maps the original labels and the original values in the aprac+aseg file.
    :param conversion_table: a table between the values of the old -> new tables. state the new voxel values
    :param bg_value: a value for voxels inside the brain we are not interested in.
    :param data_dir: that path to the Dataset.
    :return: a new Nifti1Image OR WE WILL SAVE IT HERE ALREADY AND WE WON'T RETURN ANYTHING.
    '''
    #  mask_path = os.path.join(data_dir, str(brain_id), 'MNINonLinear', consts.MASK_NAME)
    mask = nib.load(mask_path)
    # print(mask.shape, mask.get_data_dtype(), mask.affine.shape,
    #       f'mask 3D size is: 1st X 2nd X 3rd = {mask.shape[0] * mask.shape[1] * mask.shape[2]}')
    hdr = mask.header
    # print(mask.file_map, hdr.get_xyzt_units(), type(hdr), '\n', hdr.get_zooms(), '\nheader:', hdr)
    value2label_orig = {value: key for key, value in lookup_table.items()}
    voxels = mask.get_fdata()
    # print(type(voxels), voxels.shape, np.max(voxels), np.min(voxels))
    # print('length of label2value:', len(label2value))
    # return 0
    vox_copy = np.zeros([*voxels.shape, len(label2value)])
    channel_diff = min(label2value.values())
    # print(label2value.values(), channel_diff)
    # return 0

    lookup_table_values = lookup_table.values()
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            for k in range(voxels.shape[2]):
                # if i == 26 and j == 125 and k == 117:
                #     print(f'reached the bug')
                if voxels[i, j, k] not in lookup_table_values:
                    if voxels[i, j, k] == 0:
                        vox_copy[i, j, k, :] = 0
                    else:
                        vox_copy[i, j, k, label2value[label2new_label['BACKGROUND']]-channel_diff] = 1
                    # vox_copy[i, j, k, :] = 0 if voxels[i, j, k] == 0 else label2value[label2new_label['BACKGROUND']]
                else:
                    channel = label2value[  # The new value for the new label
                        label2new_label[  # The new label for the original label
                            value2label_orig[voxels[i, j, k]]]] - channel_diff # The label of the original value
                    vox_copy[i, j, k, channel] = 1
                    # print(channel)
    print(f'for debug the max value is: {vox_copy.max()}')
    new_mask = nib.nifti1.Nifti1Image(vox_copy, mask.affine, header=hdr.copy())
    return new_mask


def load_lookup_table(path: str, lookup_name: str = 'lookup-table', lr_diff=False):
    with open(path, 'r') as look:
        look_json = json.load(look)
    lookup_table = look_json[lookup_name]
    # if not lr_diff:
    #     unified
    return lookup_table


def find_values_mapping(lookup_table, new_lookup_table):
    '''
    This function finds the mappings between the values of the 2 lookup tables according to their keys.
    :param lookup_table: a dictionary of size n that its keys include the names of the new_lookup_table
     (aside from the case of minimal-lookup-table)
    :param new_lookup_table:  a dictionary of size <n that its keys are included in the names of the lookup_table
    (aside from the case of minimal-lookup-table)
    :return: a dictionary mapping the *values* of the two lookup tables according to the keys
    '''
    conversion = {}
    for key in lookup_table.keys():
        for key2 in new_lookup_table.keys():
            if key2 in key:
                '''
                        This line creates the conversion maps between the values in the lookup table and the new lookup table 
                        '''
                conversion[lookup_table[key]] = new_lookup_table[key2]
        # last_name = key.split('-')[-1]
    print(conversion)
    return conversion


def change_all_masks(label2new_label_name: str, label2value_name: str, brains_path, new_masks_path, brain_ids=None):
    if brain_ids is None:
        brain_ids = []
    count = 0
    lookup_table = load_lookup_table('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/lookupTable.json')
    label2new_label = load_lookup_table('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/label2new_label.json',
                                        lookup_name=label2new_label_name)
    label2value = load_lookup_table('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/label2value.json',
                                    lookup_name=label2value_name)

    for brain_id in brain_ids:
        brain_path = os.path.join(new_masks_path, brain_id)
        if not os.path.isdir(brain_path):
            os.mkdir(brain_path)
        brain_mask_path = os.path.join(brain_path, 'gt.nii.gz')
        if not os.path.isfile(brain_mask_path):
            mask_path = os.path.join(brains_path, brain_id, T1wConsts.MASK_NAME)
            nifti_image = update_single_multi_channel(lookup_table, label2new_label, label2value, mask_path)
            nib.nifti1.save(nifti_image, os.path.join(new_masks_path, brain_id, f'gt.nii.gz'))
            count += 1
            if count % 25 == 0:
                print(f'passed 25 currently at: {count}')
    return count

#     TODO: FINISH THE LOOP THAT ALSO SAVES THE NIFTI IMAGES


def update_main(brain_ids=None):
    if brain_ids is None:
        brain_ids = []
    count = 0
    lookup_table = load_lookup_table('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/lookupTable.json')
    unified_table = load_lookup_table('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/segmapLookupTables.json',
                                      lookup_name='unified-lookup-table')
    conversion = find_values_mapping(lookup_table, unified_table)
    # print(conversion)
    for brain_id in brain_ids:
        # update_single(brain_id, lookup_table, conversion)
        nifti_image = update_single(brain_id, lookup_table, conversion, bg_value=unified_table["BACKGROUND"])
        nib.nifti1.save(nifti_image, os.path.join(consts.UNIFIED_DIR, f'mask-{brain_id}.nii.gz'))
        print(f'created mask for brain {brain_id}')
    return count


if __name__ == '__main__':
    lookup_table_main = load_lookup_table('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/lookupTable.json')
    unified_table = load_lookup_table('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/segmapLookupTables.json',
                                      lookup_name='unified-lookup-table')
    # conversion = find_values_mapping(lookup_table_main, lookup_table_main)
    # nifti_image = update_single(100307, lookup_table_main, conversion, bg_value=2)
    # nib.nifti1.save(nifti_image, os.path.join(consts.UNIFIED_DIR, f'orig-value2mask-{100307}.nii.gz'))
    _, names = get_dataset_names(data_dir='/home/cheng/Desktop/Dataset/Generated-Masks-AGY-1.25/all-regions-and-sides-split')
    # nifti_image = update_single(lookup_table_main, label2new_label, label2value, mask_path)
    # # print(type(names[0]))
    update_main(names)
