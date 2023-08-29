import torch

TABLE_DIR = '/run/media/cheng/Maxwell_HD/Goldi_Folder/Tables/'
#  IMPORTANT: gt needs to be last because we create it after creating 'general_mask' files type.
FILES_KEYS = ['t1w', 'brain_mask', 'general_mask', 'rgb', 'gt', 'wm']
SCANS_KEYS = ['t1w', 'general_mask', 'rgb', 'gt', 'wm']
DIFFUSION_KEYS = ['dwi', 'mask', 'bvals', 'bvecs']

OLD_DATASET = '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-from-HCP/'

# THIS IS THE UNPROCESSED DATASET OF FILES STRAIGHT FROM THE HCP SITE
DATASET_07 = {'t1w': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/T1w',
              'brain_mask': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/Brain-Masks',
              'gt': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/GT',
              'general_mask': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/General-Masks',
              'rgb': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/RGB'}

# THIS IS THE PROCESSED DATASET OF FILES THAT WENT THROUGH THE CHANGE-SPACING AND GT CREATION PIPELINE.
DATASET_125 = {'t1w': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/T1w',
               'brain_mask': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Brain-Masks',
               'gt': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/GT',
               'general_mask': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/General-Masks',
               'rgb': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/RGB',
               'wm': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/WM'}

ten_pre = 16  # Tensors Precision
torch_scan_type = torch.float16
DATASET_TENSORS = {'t1w': f'/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Tensors-{ten_pre}/T1w',
                   'brain_mask': f'/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Tensors-{ten_pre}/Brain-Masks',
                   'gt': f'/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Tensors-{ten_pre}/GT',
                   'general_mask': f'/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Tensors-{ten_pre}/General-Masks',
                   'rgb': f'/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Tensors-{ten_pre}/RGB',
                   'wm': f'/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Tensors-{ten_pre}/WM'}

WM_SCANS_PATH = '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/WM'

RGB_NEW_SIZE = 'RGB-NOA-NEW-SIZE'

DIFFUSION_FILES = {'dwi': 'data.nii.gz',
                   'mask': 'nodif_brain_mask.nii.gz',
                   'bvals': 'bvals',
                   'bvecs': 'bvecs',
                   'rgb': 'RGB.nii.gz',
                   'fa': 'FA.nii.gz'}

NEW_FILES = {'t1w': 'T1w.nii.gz',
             'brain_mask': 'brain_mask.nii.gz',
             'gt': 'gt.nii.gz',
             'general_mask': 'aparc+aseg.nii.gz',
             'rgb': 'RGB.nii.gz',
             'wm': 'wm.nii.gz'}

OLD_FILES = {'t1w': 'T1w_acpc_dc_restore_brain.nii.gz',
             'brain_mask': 'brainmask_fs.nii.gz',
             'gt': 'aparc+aseg.nii.gz',
             'general_mask': 'aparc+aseg.nii.gz',
             'rgb': 'RGB.nii.gz'}

LABELS = ['CTX-LH-PARSOPERCULARIS',
          'CTX-LH-PARSTRIANGULARIS',
          'CTX-LH-SUPERIORTEMPORAL',
          'CTX-RH-PARSOPERCULARIS',
          'CTX-RH-PARSTRIANGULARIS',
          'CTX-RH-SUPERIORTEMPORAL']

VALUE_TO_CHANNEL_MAPPING = {
    1018: 1,  # CTX-LH-PARSOPERCULARIS -> Broca
    1020: 1,  # CTX-LH-PARSTRIANGULARIS -> Broca
    1030: 2,  # CTX-LH-SUPERIORTEMPORAL -> Wernicke
    2018: 1,  # CTX-RH-PARSOPERCULARIS -> Broca
    2020: 1,  # CTX-RH-PARSTRIANGULARIS -> Broca
    2030: 2  # CTX-RH-SUPERIORTEMPORAL -> Wernicke
}

VALUE_TO_CHANNEL_MAPPING_WM = {
    2: 1,  # LEFT-CEREBRAL-WHITE-MATTER
    41: 1,  # RIGHT-CEREBRAL-WHITE-MATTER
}

# before normalization
T1W_MAX = 3550.2229
T1W_MIN = 0.0
# before normalization
RGB_MAX = 1.0
RGB_MINN = 0.0

TABLE_PREFIX = 'output-table-'
NIFTI_POSTFIX = '.nii.gz'
START_ID, END_ID = len('output-table-'), -len('.txt')

# In the values for the masks we create the finest values for visualization in MRICron are BG = 10 and the rest
# 11,12,13,21,22,23. ofr visualization in WB_VIEW the best are the original labels and BG as 10 also.
