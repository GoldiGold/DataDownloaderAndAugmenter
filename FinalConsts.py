TABLE_DIR = '/run/media/cheng/Maxwell_HD/Goldi_Folder/Tables/'

FILES_KEYS = ['t1w', 'brain_masks', 'gt', 'general_masks',  'rgb']

DATASET_07 = {'t1w': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/T1w',
              'brain_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/Brain-Masks',
              'gt': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/GT',
              'general_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/General-Masks',
              'rgb': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/RGB'}

DATASET_125 = {'t1w': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/T1w',
               'brain_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Brain-Masks',
               'gt': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/GT',
               'general_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/General-Masks',
               'rgb': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/RGB'}

# SUB_DIR_DICT = {'t1w': 'T1w-0.7', 'general': 'General-Masks-0.7', 'gen': 'Generated-Masks-0.7', 'brain': 'Brain-Masks-0.7', 'rgb': 'RGB-1.25'}
# SUB_DIR_DICT_125 = {'t1w': 'T1w-1.25', 'general': 'General-Masks-1.25', 'gen': 'Generated-Masks-1.25',
#                     'brain': 'Brain-Masks-1.25', 'rgb': 'RGB-1.25'}

# DATASET_DICT = {'t1w': 'T1w', 'general': 'General-Masks', 'gen': 'Generated-Masks',
#                 'brain': 'Brain-Masks', 'rgb': 'RGB-NOA'}
RGB_NEW_SIZE = 'RGB-NOA-NEW-SIZE'
FILES = {'t1w':           {'old': 'T1w_acpc_dc_restore.nii.gz', 'new': 'T1w.nii.gz'},
         'general_masks': {'old': 'aparc+aseg.nii.gz',          'new': 'aparc+aseg.nii.gz'},
         'gt':            {'old': 'aparc+aseg.nii.gz',          'new': 'gt.nii.gz'},
         'brain_masks':   {'old': 'brainmask_fs.nii.gz',        'new': 'brain_mask.nii.gz'},
         'rgb':           {'old': 'RGB.nii.gz', 'new':          'RGB.nii.gz'}}
# MASK_NAME = {'old': 'aparc+aseg.nii.gz', 'new': 'gt.nii.gz'}
# OLD_MASK_NAME = 'aparc+aseg.nii.gz'
# MASK_NAME = 'gt.nii.gz'
# OLD_T1W_NAME = 'T1w_acpc_dc_restore.nii.gz'
# T1W_NAME = 'T1w.nii.gz'
# OLD_BRAIN_MASK_NAME = 'brainmask_fs.nii.gz'
# BRAIN_MASK_NAME = 'brain_mask.nii.gz'
# RGB_NAME = 'RGB.nii.gz'
LABELS = ['CTX-LH-PARSOPERCULARIS',
          'CTX-LH-PARSTRIANGULARIS',
          'CTX-LH-SUPERIORTEMPORAL',
          'CTX-RH-PARSOPERCULARIS',
          'CTX-RH-PARSTRIANGULARIS',
          'CTX-RH-SUPERIORTEMPORAL']

TABLE_PREFIX = 'output-table-'
NIFTI_POSTFIX = '.nii.gz'
START_ID, END_ID = len('output-table-'), -len('.txt')

# In the values for the masks we create the finest values for visualization in MRICron are BG = 10 and the rest
# 11,12,13,21,22,23. ofr visualization in WB_VIEW the best are the original labels and BG as 10 also.
