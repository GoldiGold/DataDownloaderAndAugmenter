import nibabel as nib

import os
import constsDeprecated

if __name__ == '__main__':
    mask_name = ('mask.nii.gz', 'brain_mask.nii.gz')
    brain_id = 992774
    file_dir = ('/home/cheng/Desktop/Dataset/All-Brain-Masks-1.25', '/home/cheng/Desktop/Dataset/Brain-Masks-0.7/')
    filename0 = os.path.join(file_dir[0], str(brain_id), mask_name[0])
    downsapmled_mask = nib.load(filename0)
    filename1 = os.path.join(file_dir[1], str(brain_id), mask_name[1])
    regular_mask = nib.load(filename1)
    filename2 = os.path.join(file_dir[0], str(brain_id), mask_name[1])
    noa_mask = nib.load(filename2)
    noa_x_down = nib.load('/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/noaXdown.nii.gz')
    for att0, att1 in zip(noa_x_down.header, noa_mask.header):
        print(f'{att0}: {noa_x_down.header[att0]}, {att1}: {noa_mask.header[att1]}')

    # nib.save(nib.Nifti1Image(downsapmled_mask.get_fdata(), noa_mask.affine, header=noa_mask.header),
    #          '/home/cheng/Desktop/Dataset/noaXdown.nii.gz')
