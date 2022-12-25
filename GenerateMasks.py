import os
import json
import ChangeMaskFiles
import consts


def generate_masks(brains_path: str, new_masks_path: str, mode: str):
    brain_ids = sorted([str(i) for i in os.listdir(brains_path) if i.isdigit()])  # if this is the correct syntax
    if mode == 'first':
        # read json file label 2 new label
        label2new_label = 'all regions and sides split'
        # read json file label 2 value
        label2value = 'all regions and sides split'
        mask_name = '1st mask'
    elif mode == 'second':
        # read json file label 2 new label
        label2new_label = 'all regions split'
        # read json file label 2 value
        label2value = 'all regions split'
        mask_name = '2nd mask'
    elif mode == 'third':
        # read json file label 2 new label
        label2new_label = 'all sides split'
        # read json file label 2 value
        label2value = 'all sides split'
        mask_name = '3rd mask'
    elif mode == 'fourth':
        # read json file label 2 new label
        label2new_label = 'minimal split'
        # read json file label 2 value
        label2value = 'minimal split'
        mask_name = '4th mask'
    else:
        # Set as default to the first case of split all.
        # read json file label 2 new label
        label2new_label = 'all regions and sides split'
        # read json file label 2 value
        label2value = 'all regions and sides split'
        mask_name = '1st mask'
        # masks_counter = 0
        # existed_counter = 0
        # for idx in indices:
        # 	mask_dir = os.path.join(new_masks_path, subdir, idx)
        # 	if !os.path.exists(mask_dir):
        # 		os.mkdir(mask_dir)
        # 	if !os.path.exists(os.path.join(mask_dir, mask_name)):
        # 		#enter mask creation function from our code with label -> new label -> value
        # 		masks_counter += 1
        # 	else:
        # 		existed_counter += 1
        # brains_path = os.path.join(consts.SSD_DATASET, 'Generate-Maks')
    subdir = os.path.join(new_masks_path, label2new_label)

    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    masks_created = ChangeMaskFiles.change_all_masks(label2new_label_name=label2new_label, label2value_name=label2value,
                                                     brains_path=brains_path, new_masks_path=subdir,
                                                     brain_ids=brain_ids[1:])
    return masks_created


if __name__ == '__main__':
    m_c = generate_masks(brains_path='/home/cheng/Desktop/Dataset/Dataset-1.25/General-Masks/',
                         new_masks_path='/home/cheng/Desktop/Dataset/Dataset-1.25/Generated-Masks/',
                         mode='fourth')
    print(f'mask created {m_c}')
