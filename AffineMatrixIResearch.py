import nibabel as nib
import os


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

# def func(brain: nib.Nifti1Image):
    # brain.get_fdata()

if __name__ == '__main__':
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
