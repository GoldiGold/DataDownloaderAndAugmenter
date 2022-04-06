import consts
import os
import shutil


def move_T1w():
    '''
    This function transfer only the T1w_restore_brain.nii.gz files of the brains from the Hard drive to the ssd for fast
    reading and writing
    :return:
    '''
    return 1


def move_aparc():
    '''
    This function transfer only the aparc+aseg.nii.gz files of the brains from the Hard drive to the ssd for fast
    reading and writing
    :return:
    '''
    return 2


def move_only_broca_wernike_brains():
    '''
    This function transfer only the nifti files that have only broca and wernike areas segmented with our editing
    function from the Hard drive to the ssd for fast reading and writing
    :return:
    '''
    return 3


def move_mri_files(indices: list, dst_dir: str, file_name: str):
    # src_dir = f'/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset/{brain_index}/MNINonLinear/'
    for brain_index in indices:
        src_file = os.path.join(consts.DATASET_DIR, brain_index, 'MNINonLinear', file_name)
        dst_file = os.path.join(dst_dir, brain_index, file_name)
        if not os.path.isdir(os.path.join(dst_dir, brain_index)):
            os.mkdir(os.path.join(dst_dir, brain_index))
        if not os.path.isfile(dst_file):
            shutil.copy(src_file, dst_file)


# def get
def move_file(file_name: str, dst_dir):
    return 1


def main():
    ssd_t1w = os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT['t1w'])
    idxs = [100307]
    move_mri_files([str(i) for i in idxs], ssd_t1w, 'T1w_restore_brain.nii.gz')
    return 1


if __name__ == '__main__':
    ssd_t1w = os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT['t1w'])
    ssd_general = os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT['general'])
    ssd_gen = os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT['gen'])
    main()
