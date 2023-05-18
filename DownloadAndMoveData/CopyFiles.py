import os
import shutil
import consts
import T1wConsts
import nibabel as nib
# import torch

t1w_src_path_suffix = f'MNINonLinear/{consts.T1W_NAME}'
t1w_dst_path_suffix = f'{consts.T1W_NAME}'
mask_src_path_suffix = f'MNINonLinear/{consts.MASK_NAME}'
mask_dst_path_suffix = f'{consts.MASK_NAME}'
brain_masks_src_path = f'MNINonLinear/{consts.OLD_BRAIN_NAME}'
brain_masks_dst_path = f'{consts.BRAIN_NAME}'

def save_as_tensors(brain_ids: list, brains_path: str, tensors_path: str, brain_file_name: str):
    for brain in brain_ids:
        brain_data = nib.load(os.path.join(brains_path, brain, brain_file_name)).get_fdata()
        # brain_tensor = torch.save(os.path.join(tensors_path, f'{brain}.pt'))

def copy_brain_masks(src_dataset: str, dst_dataset: str):
    indices = [str(i) for i in os.listdir(src_dataset) if i.isdigit()]
    copied_counter = 0
    failed_counter = 0
    for idx in indices:
        src_file = os.path.join(src_dataset, idx, brain_masks_src_path)
        path_dst = os.path.join(dst_dataset, 'Brain-Masks', idx)
        if os.path.exists(src_file) and os.path.isfile(src_file):
            if not os.path.exists(path_dst):
                os.mkdir(path_dst)
            if os.path.isdir(path_dst):
                dst_file = os.path.join(path_dst, brain_masks_dst_path)
                if not os.path.exists(dst_file):
                    shutil.copy(src_file, dst_file)
                    copied_counter += 1
        else:
            print(f'ERROR couldnt copy t1w of brain number {idx}')
            failed_counter += 1
        # break  # TO check for the first brain
    print(
        f't1w: {copied_counter}, failed t1w: {failed_counter}')
    print(f'amount of brains to copy: {len(indices)}')


def copy_masks(src_dataset: str, dst_dataset: str):
    indices = [str(i) for i in os.listdir(src_dataset) if i.isdigit()]
    copied_t1w_counter = 0
    copied_masks_counter = 0
    failed_t1w_counter = 0
    failed_masks_counter = 0
    for idx in indices:
        t1w_src_file = os.path.join(src_dataset, idx, t1w_src_path_suffix)
        t1w_path_dst = os.path.join(dst_dataset, 'T1w', idx)
        if os.path.exists(t1w_src_file) and os.path.isfile(t1w_src_file):
            if not os.path.exists(t1w_path_dst):
                os.mkdir(t1w_path_dst)
            if os.path.isdir(t1w_path_dst):
                t1w_dst_file = os.path.join(t1w_path_dst, t1w_dst_path_suffix)
                if not os.path.exists(t1w_dst_file):
                    shutil.copy(t1w_src_file, t1w_dst_file)
                    copied_t1w_counter += 1
        else:
            print(f'ERROR couldnt copy t1w of brain number {idx}')
            failed_t1w_counter += 1
        mask_src_file = os.path.join(src_dataset, idx, mask_src_path_suffix)
        mask_path_dst = os.path.join(dst_dataset, 'General-Masks', idx)
        if os.path.exists(mask_src_file) and os.path.isfile(mask_src_file):
            if not os.path.exists(mask_path_dst):
                os.mkdir(mask_path_dst)
            if os.path.isdir(mask_path_dst):
                mask_dst_file = os.path.join(mask_path_dst, mask_dst_path_suffix)
                if not os.path.exists(mask_dst_file):
                    shutil.copy(mask_src_file, mask_dst_file)
                    copied_masks_counter += 1
        else:
            print(f'ERROR couldnt copy mask of brain number {idx}')
            failed_masks_counter += 1
        # break  # TO check for the first brain
    print(
        f't1w: {copied_t1w_counter}, masks: {copied_masks_counter}, failed t1w: {failed_t1w_counter}, failed masks: {failed_masks_counter}')
    print(f'amount of brains to copy: {len(indices)}')


if __name__ == '__main__':
    src_dataset = consts.DATASET_DIR
    dst_dataset = consts.SSD_DATASET
    copy_brain_masks(src_dataset=src_dataset, dst_dataset=dst_dataset)
