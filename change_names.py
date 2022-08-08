import os
import consts


def remove_suffix_from_names(directory_path: str, suffix: str):
    names = os.listdir(directory_path)
    for name in names:
        if name[-len(suffix):] == suffix:
            new_name = name[:-len(suffix)]
            os.rename(os.path.join(directory_path, name), os.path.join(directory_path, new_name))


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
    # main_brain_name_changes(os.path.join(consts.SSD_DATASET, consts.SUB_DIR_DICT_125['gen']), 'mask-1.25.nii.gz',
      #                      'mask.nii.gz')
    remove_suffix_from_names('/home/cheng/Desktop/Dataset/Dataset-2.5', '-2.5')
