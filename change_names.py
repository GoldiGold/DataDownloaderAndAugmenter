import os
import consts


def change_name(new: str, old: str):
    if os.path.isfile(new):
        return 1
    if not os.path.isfile(old) and not os.path.isfile(new):
        raise ValueError(f"{old} is not a file and {new} doesn't exist")
    else:
        os.renames(old, new)


def main_brain_name_changes(dirname: str, old_names: str, new_names: str):
    brain_ids = sorted([str(i) for i in os.listdir(dirname) if i.isdigit()])  # if this is the correct syntax

    for brain in brain_ids:
        change_name(os.path.join(dirname, brain, new_names), os.path.join(dirname, brain, old_names))


if __name__ == '__main__':
    main_brain_name_changes(os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT_125['gen']), 'mask-1.25.nii.gz',
                           'mask.nii.gz')
