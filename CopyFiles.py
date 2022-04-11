import os
import shutil
import consts

t1w_src_path_suffix = f'MNINonLinear/{consts.T1W_NAME}'
t1w_dst_path_suffix = f'{consts.T1W_NAME}'
mask_src_path_suffix = f'MNINonLinear/{consts.MASK_NAME}'
mask_dst_path_suffix = f'{consts.MASK_NAME}'


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
    copy_masks(src_dataset=src_dataset, dst_dataset=dst_dataset)
