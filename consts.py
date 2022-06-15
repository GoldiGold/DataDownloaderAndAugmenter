DATA_DIR = '/run/media/cheng/Maxwell_HD/Goldi_Folder/'
UNIFIED_DIR = '/run/media/cheng/Maxwell_HD/Goldi_Folder/UnifiedMasks'
DATASET_DIR = '/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset'
WB_DIR = '/home/cheng/Desktop/workbench/exe_linux64/'
TABLE_DIR = '/run/media/cheng/Maxwell_HD/Goldi_Folder/Tables/'
SSD_DATASET = '/home/cheng/Desktop/Dataset/'
SUB_DIR_DICT = {'t1w': 'T1w-0.7', 'general': 'General-Masks-0.7', 'gen': 'Generated-Masks-0.7', 'brain': 'Brain-Masks-0.7'}
SUB_DIR_DICT_125 = {'t1w': 'T1w-1.25', 'general': 'General-Masks-1.25', 'gen': 'Generated-Masks-1.25',
                    'brain': 'Brain-Masks-1.25'}

OLD_MASK_NAME = 'aparc+aseg.nii.gz'
MASK_NAME = 'mask.nii.gz'
OLD_T1W_NAME = 'T1w_restore_brain.nii.gz'
T1W_NAME = 'T1w.nii.gz'
OLD_BRAIN_NAME = 'brainmask_fs.nii.gz'
BRAIN_NAME = 'brain_mask.nii.gz'
LABELS = ['CTX-LH-PARSOPERCULARIS', 'CTX-LH-PARSTRIANGULARIS', 'CTX-LH-SUPERIORTEMPORAL', 'CTX-RH-PARSOPERCULARIS',
          'CTX-RH-PARSTRIANGULARIS', 'CTX-RH-SUPERIORTEMPORAL']
TABLE_PREFIX = 'output-table-'
NIFTI_POSTFIX = '.nii.gz'
START_ID, END_ID = len('output-table-'), -len('.txt')

# In the values for the masks we create the finest values for visualization in MRICron are BG = 10 and the rest
# 11,12,13,21,22,23. ofr visualization in WB_VIEW the best are the original labels and BG as 10 also.
