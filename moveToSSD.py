import consts, os


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


def main():
    return 1


if __name__ == '__main__':
    main()
