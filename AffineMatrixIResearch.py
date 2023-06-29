import nibabel as nib
import torch
import os
import FinalConsts


def check_affine_matrix_polarity(path: str):
    brain = nib.load(path)
    aff = brain.affine
    dist = 0
    angle = 0
    if aff[0][3] < 0 or aff[1][3] < 0 or aff[2][3] < 0:
        dist = 1
    if aff[0][0] < 0 or aff[1][1] < 0 or aff[2][2] < 0:
        angle = 1
    return dist, angle
    # data = brain.get_fdata()
    # print(aff, data.shape)


def count_negative_polarities():
    dist_negative_counter, angle_negative_counter = 0, 0
    brain_dir = "/media/chen/Maxwell_HD/Computer_Files/home/cheng/Desktop/Dataset/T1w/"
    brains = os.listdir(brain_dir)
    for brain in brains:
        negative_counter = check_affine_matrix_polarity(os.path.join(brain_dir, brain, "T1w.nii.gz"))
        dist_negative_counter += negative_counter[0]
        angle_negative_counter += negative_counter[1]
    # check_affine_matrix_polarity("/media/chen/Maxwell_HD/Computer_Files/home/cheng/Desktop/Dataset/T1w/100206/T1w.nii.gz")
    print(f'out of {len(brains)} brains {dist_negative_counter} have negative value in the dist columns')
    print(f'out of {len(brains)} brains {angle_negative_counter} have negative value in the angle blocks')


def research_polarities():
    new_brain = '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/T1w/100206/T1w.nii.gz'
    old_brain = '/media/chen/Maxwell_HD/Computer_Files/home/cheng/Desktop/Dataset/T1w/100206/T1w.nii.gz'
    super_old_brain = '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/T1w/100206/T1w.nii.gz'

    new_scan = nib.load(new_brain)
    old_scan = nib.load(old_brain)
    super_old_scan = nib.load(super_old_brain)

    print('new affine:')
    print(new_scan.affine)

    print('old affine:')
    print(old_scan.affine)

    print('super old affine:')
    print(super_old_scan.affine)

    count_negative_polarities()

def func(brain: nib.Nifti1Image):
    brain.get_fdata()


def research_hadassa():
    t1w_scan = nib.load('/home/chen/Downloads/HadasaBrainData/T1_MNI_125mm_extracted.nii.gz')
    rgb_scan = nib.load('/home/chen/Downloads/HadasaBrainData/RGB.nii.gz')
    t1w_data = t1w_scan.get_fdata()
    rgb_data = rgb_scan.get_fdata()
    print(f't1w data shape: {t1w_data.shape}\n t1w affine: \n{t1w_scan.affine}')
    print(f'rgb data shape: {rgb_data.shape}\n rgb affine: \n{rgb_scan.affine}')

#     check a brain from the dataset
    dataset_scan = nib.load(os.path.join('/media/chen/Maxwell_HD/Computer_Files/home/cheng/Desktop/Dataset/T1w/100206/T1w.nii.gz'))
    dataset_data = dataset_scan.get_fdata()
    print(f'from dataset data shape: {dataset_data.shape}\n from dataset affine: \n{dataset_scan.affine}')

    tensor_dataset = torch.load('/media/chen/Maxwell_HD/Computer_Files/home/cheng/Desktop/Dataset/T1w-Tensors/100206.pt')
    print(tensor_dataset.shape)
    tensor_dataset = torch.load(
        '/media/chen/Maxwell_HD/Computer_Files/home/cheng/Desktop/Dataset/RGB-Tensors/100206.pt')
    print(tensor_dataset.shape)


if __name__ == '__main__':
    research_hadassa()
