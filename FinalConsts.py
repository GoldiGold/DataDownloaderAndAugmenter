TABLE_DIR = '/run/media/cheng/Maxwell_HD/Goldi_Folder/Tables/'

FILES_KEYS = ['t1w', 'brain_masks', 'gt', 'general_masks',  'rgb']

OLD_DATASET = '/media/chen/Maxwell_HD/Goldi_Folder/Dataset/'

# THIS IS THE UNPROCESSED DATASET OF FILES STRAIGHT FROM THE HCP SITE
DATASET_07 = {'t1w': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/T1w',
              'brain_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/Brain-Masks',
              'gt': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/GT',
              'general_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/General-Masks',
              'rgb': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/0.7/RGB'}

# THIS IS THE PROCESSED DATASET OF FILES THAT WENT THROUGH THE CHANGE-SPACING AND GT CREATION PIPELINE.
DATASET_125 = {'t1w': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/T1w',
               'brain_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/Brain-Masks',
               'gt': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/GT',
               'general_masks': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/General-Masks',
               'rgb': '/media/chen/Maxwell_HD/Goldi_Folder/Dataset-T1w/1.25/RGB'}

RGB_NEW_SIZE = 'RGB-NOA-NEW-SIZE'
FILES = {'t1w':           {'old': 'T1w_acpc_dc_restore.nii.gz', 'new': 'T1w.nii.gz'},
         'general_masks': {'old': 'aparc+aseg.nii.gz',          'new': 'aparc+aseg.nii.gz'},
         'gt':            {'old': 'aparc+aseg.nii.gz',          'new': 'gt.nii.gz'},
         'brain_masks':   {'old': 'brainmask_fs.nii.gz',        'new': 'brain_mask.nii.gz'},
         'rgb':           {'old': 'RGB.nii.gz', 'new':          'RGB.nii.gz'}}

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
