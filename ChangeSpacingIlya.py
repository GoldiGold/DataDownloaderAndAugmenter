# built in
import os
import multiprocessing

# 3rd party
import numpy as np
import nibabel as nib

# dipy
import dipy.reconst.dti as dti
from dipy.io import read_bvals_bvecs
from dipy.align.imaffine import AffineMap
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa

# misc
import consts

# from utils import tic, toc


data_dir = os.path.join(consts.SSD_DATASET)
# consts.SUB_DIR_DICT['t1w'])  # r'/run/media/noa/DATA1/noab/cilabCode/data/dataset_hcp105_raw'
set_type = r'32g_25mm'  # Diffusion / 32g_25mm
num_procs = 1  # Multiprocessing mode
case_list = r'hcp105_list.txt'
generated_masks_type = 'all regions and sides split'

mask_sizes = {'HCP': (260, 311, 260), 'HARDI': (145, 174, 145), 'HARDI_NOA': (128, 144, 128), 'CLINICAL': (73, 87, 73)}


def change_spacing_4D(img_in, new_spacing=1.25):
    # data = img_in.get_data()

    # img_in = nib.load(os.path.join(data_dir, img_id, consts.T1W_NAME))
    data = img_in.get_fdata()
    image_3d = False
    if data.ndim == 3:
        image_3d = True
        data = data[..., np.newaxis]

    old_shape = data.shape
    img_spacing = abs(img_in.affine[0, 0])

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)
    new_affine[0, 0] = new_spacing if img_in.affine[0, 0] > 0 else -new_spacing
    new_affine[1, 1] = new_spacing if img_in.affine[1, 1] > 0 else -new_spacing
    new_affine[2, 2] = new_spacing if img_in.affine[2, 2] > 0 else -new_spacing

    if new_spacing == 1.25:
        new_shape = mask_sizes['HARDI']
    elif new_spacing == 1.5:
        new_shape = mask_sizes['HARDI_NOA']
    elif new_spacing == 2.5:
        new_shape = mask_sizes['CLINICAL']
    else:
        # original version
        new_shape = np.floor(np.array(img_in.get_fdata().shape) * (img_spacing / new_spacing))
    new_shape = new_shape[:3]  # drop last dim

    new_data = []
    for i in range(data.shape[3]):
        affine_map = AffineMap(np.eye(4),
                               new_shape, new_affine,
                               old_shape, img_in.affine
                               )
        # Generally nearest a bit better results than linear interpolation
        # res = affine_map.transform(data[:,:,:,i], interp="linear")
        res = affine_map.transform(data[:, :, :, i], interpolation="nearest")
        new_data.append(res)

    new_data = np.array(new_data).transpose(1, 2, 3, 0)

    if image_3d:
        new_data = new_data[..., 0]

    img_new = nib.Nifti1Image(new_data.astype(data.dtype), new_affine)

    return img_new


def temp_try_t1w_with_brain():
    in_files_dict = {
        't1': '/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset/992774/MNINonLinear/T1w_restore_brain.nii.gz',
        'brain_mask': '/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset/992774/MNINonLinear/brainmask_fs.nii.gz',
        'generated_mask': None, 'general_mask': None, 'rgb': None}
    out_files_dict = {
        't1': 'T1w_restore_brain_1.25.nii.gz',
        'brain_mask': 'brainmask_fs_1.25.nii.gz',
        'generated_mask': None, 'general_mask': None, 'rgb': None}
    update_single_brain(in_files_dict, out_files_dict, 992774, 1.25)


def update_single_brain(in_files_dict: dict, out_files_dict: dict, case_idx, new_spacing=1.25):
    '''
    gets the in files names and the out files names in dictionaries and creates the new files with the new spacing
    that has the keys:

    :param in_files_dict: holds the full path to the files we alter.
        keys: 't1', 'brain_mask', 'generated_mask', 'general_mask', 'rgb'
    :param out_files_dict: holds the full path to the files we create.
        keys: 't1', 'brain_mask', 'generated_mask', 'general_mask', 'rgb'
    :param new_spacing: the new resolution of the scans that we give the function 'change_spacing_4D'
    :return: 0 if didn't do anything (no brain mask), and nothing - prints a success message
    '''
    if not os.path.isfile(in_files_dict['brain_mask']):
        print('>>> processed case: {} NO BRAIN MASK - DID NOT CHANGED SIZES'.format(case_idx))
        return 0

    # # Generate DTI
    # print('Load DWI data ... ', end='')
    # img = nib.load(dwi_file)
    # data = img.get_data()
    # print(data.shape, data.dtype)

    print('Load mask ... ', end='')
    mask_hq = nib.load(
        in_files_dict['brain_mask'])  # it was brain_mask before which is the binary map of where the brain
    # is in the skull check and see if it is the same as our brain_mask files or maybe they are bigger
    # (0.7 mm of precision instead of 1.25)
    mask_lq = mask_hq.get_fdata()

    mask_lq = change_spacing_4D(mask_hq, new_spacing=new_spacing)

    mask_lq_data = mask_lq.get_fdata()
    nib.save(nib.Nifti1Image(mask_lq_data.astype(np.float32), mask_lq.affine), out_files_dict['brain_mask'])

    for file_key in in_files_dict.keys():
        # for in_file, out_file in zip([rgb_file_in],
        #                              [rgb_file_out]):
        if in_files_dict[file_key] is None or not os.path.isfile(in_files_dict[file_key]):
            continue
        scan = nib.load(in_files_dict[file_key])
        altered_scan = change_spacing_4D(scan, new_spacing=new_spacing)
        altered_scan_data = altered_scan.get_fdata()
        altered_scan_data[altered_scan_data < 0] = 0
        # print('Before processing: {}, min: {}, max: {}, shape: {}'.format(altered_scan.dtype, altered_scan.min(), altered_scan.max(), altered_scan.shape))
        if len(altered_scan_data.shape) == 4:
            for dim in range(altered_scan_data.shape[3]):
                altered_scan_data[:, :, :, dim] = altered_scan_data[:, :, :, dim] * mask_lq_data
        else:
            altered_scan_data = altered_scan_data * mask_lq_data
        # print('After processing: {}, min: {}, max: {}, shape: {}'.format(altered_scan.dtype, altered_scan.min(), altered_scan.max(), altered_scan.shape))
        altered_scan_img = nib.Nifti1Image(altered_scan_data.astype(np.float32), altered_scan.affine)
        nib.save(altered_scan_img, out_files_dict[file_key])

    print('>>> processed case: {}'.format(case_idx))


def update_goldi_files(case_idx, new_spacing=1.25):
    # case_dir = os.path.join(data_dir, case)
    t1_file_in = os.path.join(data_dir, consts.SUB_DIR_DICT['t1w'], case_idx, consts.T1W_NAME)
    brain_mask_file = os.path.join(data_dir, consts.SUB_DIR_DICT['brain'], case_idx, consts.BRAIN_NAME)
    gen_mask_file = os.path.join(data_dir, consts.SUB_DIR_DICT['gen'], generated_masks_type, case_idx, consts.MASK_NAME)
    reg_mask_file = os.path.join(data_dir, consts.SUB_DIR_DICT['general'], case_idx, consts.OLD_MASK_NAME)
    rgb_file_in = os.path.join(data_dir, consts.SUB_DIR_DICT['rgb'], case_idx, consts.RGB_NAME)

    if not os.path.isfile(brain_mask_file):
        print('>>> processed case: {} NO BRAIN MASK - DID NOT CHANGED SIZES'.format(case_idx))
        return 0
    t1_out_dir = os.path.join(data_dir, f'T1w-AGY-{new_spacing}', case_idx)
    gen_mask_out_dir = os.path.join(data_dir, f'Generated-Masks-AGY-{new_spacing}', case_idx)
    reg_mask_out_dir = os.path.join(data_dir, f'General-Masks-AGY-{new_spacing}', case_idx)
    brain_mask_out_dir = os.path.join(data_dir, f'Brain-Masks-AGY-{new_spacing}', case_idx)
    rgb_out_dir = os.path.join(data_dir, f'RGB-AGY-{new_spacing}', case_idx)

    t1_file_out = os.path.join(t1_out_dir, 'T1w.nii.gz')
    gen_mask_file_out = os.path.join(gen_mask_out_dir, 'mask.nii.gz')
    reg_mask_file_out = os.path.join(reg_mask_out_dir, 'aparc+aseg.nii.gz')
    brain_mask_file_out = os.path.join(brain_mask_out_dir, 'brain_mask.nii.gz')
    rgb_file_out = os.path.join(rgb_out_dir, 'RGB.nii.gz')

    for dirs in [t1_out_dir, gen_mask_out_dir, reg_mask_out_dir, brain_mask_out_dir, rgb_out_dir]:
        os.makedirs(dirs, exist_ok=True)

    # # Generate DTI
    # print('Load DWI data ... ', end='')
    # img = nib.load(dwi_file)
    # data = img.get_data()
    # print(data.shape, data.dtype)

    print('Load mask ... ', end='')
    mask_hq = nib.load(
        brain_mask_file)  # it was brain_mask before which is the binary map of where the brain is in the skull
    # check and see if it is the same as our brain_mask files or maybe they are bigger
    # (0.7 mm of precision instead of 1.25)
    mask_lq = mask_hq.get_fdata()
    # if set_type == '32g_25mm':
    mask_lq = change_spacing_4D(mask_hq, new_spacing=new_spacing)
    # mask_lq_file = os.path.join(out_dir, 'brain_mask.nii.gz')
    # nib.save(mask_lq, mask_lq_file)
    mask_lq_data = mask_lq.get_fdata()
    nib.save(nib.Nifti1Image(mask_lq_data.astype(np.float32), mask_lq.affine), brain_mask_file_out)
    # maskdata = data * mask_lq[..., np.newaxis]
    # print(maskdata.shape, maskdata.dtype)

    # Process T1 - BUT THIS SEEMS TO NOT REDUCE THE PRECISION FROM 0.7 MM TO 1.25 MM SO WE NEED TO PUT IT
    # THROUGH THE PIPLINE

    for in_file, out_file in zip([t1_file_in, gen_mask_file, reg_mask_file],
                                 [t1_file_out, gen_mask_file_out, reg_mask_file_out]):
        # for in_file, out_file in zip([rgb_file_in],
        #                              [rgb_file_out]):
        scan = nib.load(in_file)
        altered_scan = change_spacing_4D(scan, new_spacing=new_spacing)
        altered_scan_data = altered_scan.get_fdata()
        altered_scan_data[altered_scan_data < 0] = 0
        # print('Before processing: {}, min: {}, max: {}, shape: {}'.format(altered_scan.dtype, altered_scan.min(), altered_scan.max(), altered_scan.shape))
        if len(altered_scan_data.shape) == 4:
            for dim in range(altered_scan_data.shape[3]):
                altered_scan_data[:, :, :, dim] = altered_scan_data[:, :, :, dim] * mask_lq_data
        else:
            altered_scan_data = altered_scan_data * mask_lq_data
        # print('After processing: {}, min: {}, max: {}, shape: {}'.format(altered_scan.dtype, altered_scan.min(), altered_scan.max(), altered_scan.shape))
        altered_scan_img = nib.Nifti1Image(altered_scan_data.astype(np.float32), altered_scan.affine)
        nib.save(altered_scan_img, out_file)

    # print('Load T1 data ... ')
    # t1 = nib.load(t1_file_in)
    # T1 = change_spacing_4D(t1, new_spacing=1.25).get_fdata()
    # T1[T1 < 0] = 0
    # print('Before processing: {}, min: {}, max: {}, shape: {}'.format(T1.dtype, T1.min(), T1.max(), T1.shape))
    # T1 = T1 * mask_lq
    # print('After processing: {}, min: {}, max: {}, shape: {}'.format(T1.dtype, T1.min(), T1.max(), T1.shape))
    # t1_img = nib.Nifti1Image(T1.astype(np.float32), t1.affine)
    # print('Saving T1: ', t1_file_out)
    # nib.save(t1_img, t1_file_out)

    # case_time = toc()

    print('>>> processed case: {}'.format(case_idx))


def case_cb(case_idx, case):
    tic()

    case_dir = os.path.join(data_dir, case)
    t1_file_in = os.path.join(case_dir, 'T1w_acpc_dc_restore_1.25.nii.gz')
    mask_file = os.path.join(case_dir, 'nodif_brain_mask.nii.gz')

    if set_type == 'Diffusion':
        out_dir = os.path.join(case_dir, 'HARDI_DIPY')
        dwi_file = os.path.join(case_dir, set_type, 'data.nii.gz')
        bval_file = os.path.join(case_dir, set_type, 'bvals')
        bvec_file = os.path.join(case_dir, set_type, 'bvecs')
    elif set_type == '32g_25mm':
        out_dir = os.path.join(case_dir, 'CLINICAL_DIPY')
        dwi_file = os.path.join(case_dir, set_type, 'Diffusion.nii.gz')
        bval_file = os.path.join(case_dir, set_type, 'Diffusion.bvals')
        bvec_file = os.path.join(case_dir, set_type, 'Diffusion.bvecs')
    else:
        raise ValueError('Invalid set type {} (should be Diffusion or 32g_25mm)'.format(set_type))

    fa_file = os.path.join(out_dir, 'FA.nii.gz')
    rgb_file = os.path.join(out_dir, 'RGB.nii.gz')
    pdd_file = os.path.join(out_dir, 'PDD.nii.gz')
    t1_file_out = os.path.join(out_dir, 'T1.nii.gz')

    os.makedirs(out_dir, exist_ok=True)

    # Generate DTI
    print('Load DWI data ... ', end='')
    img = nib.load(dwi_file)
    data = img.get_data()
    print(data.shape, data.dtype)

    print('Load mask ... ', end='')
    mask_hq = nib.load(mask_file)
    mask_lq = mask_hq.get_data()
    if set_type == '32g_25mm':
        mask_lq = change_spacing_4D(mask_hq, new_spacing=2.5)
        # mask_lq_file = os.path.join(out_dir, 'brain_mask.nii.gz')
        # nib.save(mask_lq, mask_lq_file)
        mask_lq = mask_lq.get_data()
    maskdata = data * mask_lq[..., np.newaxis]
    print(maskdata.shape, maskdata.dtype)

    print('Load BVALs & BVECs data ... ')
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals, bvecs)

    print('Fitting DTI model ... ')
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    print('Compute FA ... ', end='')
    FA = fractional_anisotropy(tenfit.evals)
    print(FA.shape, FA.dtype)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
    if set_type == '32g_25mm':
        fa_img = change_spacing_4D(fa_img, new_spacing=1.25)
    print('Saving FA: ', fa_file)
    nib.save(fa_img, fa_file)

    PDD = tenfit.evecs[..., 0]
    pdd_img = nib.Nifti1Image(PDD.astype(np.float32), img.affine)
    if set_type == '32g_25mm':
        pdd_img = change_spacing_4D(pdd_img, new_spacing=1.25)
    print('Saving PDD: ', pdd_file)
    nib.save(pdd_img, pdd_file)

    print('Compute RGB ... ', end='')
    RGB = color_fa(FA, tenfit.evecs)

    rgb_img = nib.Nifti1Image(RGB.astype(np.float32), img.affine)
    if set_type == '32g_25mm':
        rgb_img = change_spacing_4D(rgb_img, new_spacing=1.25)
    print('Saving RGB: ', rgb_file)
    nib.save(rgb_img, rgb_file)

    # Process T1
    print('Load T1 data ... ')
    t1 = nib.load(t1_file_in)
    T1 = t1.get_data()
    T1[T1 < 0] = 0
    print('Before processing: {}, min: {}, max: {}'.format(T1.dtype, T1.min(), T1.max()))
    T1 *= mask_hq.get_data()
    print('After processing: {}, min: {}, max: {}'.format(T1.dtype, T1.min(), T1.max()))
    t1_img = nib.Nifti1Image(T1.astype(np.float32), t1.affine)
    print('Saving T1: ', t1_file_out)
    nib.save(t1_img, t1_file_out)

    case_time = toc()

    print('>>> processed case: {} ({}) in {:.2f} [sec]'.format(case, case_idx + 1, case_time / 1000))


if __name__ == "__main__":
    temp_try_t1w_with_brain()

    '''
    brain_ids = sorted([str(i) for i in os.listdir(os.path.join(data_dir, consts.SUB_DIR_DICT['t1w'])) if
                        i.isdigit()])  # if this is the correct syntax
    # cases = [line.rstrip() for line in fp.readlines()]

    if num_procs <= 1:
        # Serial execution
        for case_idx in ['992774']:  # brain_ids[:1]:
            update_goldi_files(case_idx, 1.25)
    else:
        # Parallel execution
        jobs = []
        for case_idx in brain_ids[:40]:
            p = multiprocessing.Process(target=update_goldi_files, args=tuple(case_idx, 2.5))
            jobs.append(p)

        num_batches = int(np.ceil(len(jobs) / num_procs))
        for batch in range(num_batches):
            batch_jobs = jobs[num_procs * batch: min(num_procs * (batch + 1), len(jobs))]
            for job in batch_jobs:
                job.start()

            for job in batch_jobs:
                job.join()
    '''
