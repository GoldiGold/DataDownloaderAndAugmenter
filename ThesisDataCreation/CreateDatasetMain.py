from CreateNewSpacingDataset import *
from FinalConsts import DATASET_07, DATASET_125


if __name__ == '__main__':
    # create_dataset(DATASET_07, DATASET_125, 1.25)
    # create_all_scans_for_dataset(DATASET_07, DATASET_125, DATASET_125['brain_mask'])
    create_all_rgb_and_fa_scans('/media/chen/Passport Sheba/HCP-Diffusion-Files/Diffusion-Files/',
                                DATASET_125['rgb'], DATASET_125['brain_mask'])
