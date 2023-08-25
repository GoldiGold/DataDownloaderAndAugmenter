from CreateNewSpacingDataset import *
from FinalConsts import DATASET_07, DATASET_125, DATASET_TENSORS
from CreateFilesGT import *
from CreateDataTensors import convert_all_scans_to_tensors

if __name__ == '__main__':
    # create_dataset(DATASET_07, DATASET_125, 1.25)
    # create_all_scans_for_dataset(DATASET_07, DATASET_125, DATASET_125['brain_mask'])
    create_all_rgb_and_fa_scans('/media/chen/Passport Sheba/HCP-Diffusion-Files/Diffusion-Files/',
                                DATASET_125['rgb'], DATASET_125['brain_mask'])
    # create_all_gt_scans(DATASET_125['general_mask'], DATASET_125['gt'])
    # convert_all_scans_to_tensors(DATASET_125, DATASET_TENSORS)
