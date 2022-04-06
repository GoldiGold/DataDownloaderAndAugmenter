import os
import shutil
t1w_src_path_suffix = 'enter the real one'
t1w_dst_path_suffix = 'enter the real one'

def copy_masks(src_dataset: str, dst_dataset: str):
	indices = [str(i) for i in os.listdir(src_dataset) if i.isdigit()]
	for idx in indices:
		t1w_src_file = os.path.join(src_dataset, idx, t1w_src_path_suffix)
		t1w_path_dst = os.path.join(dst_dataset, 'T1w', idx)
		if os.path.exists(t1w_src_file) and os.path.isfile(t1w_src_file):
			if os.path.isdir(t1w_path_dst):
				t1w_dst_file = os.path.join(t1w_path_dst, t1w_dst_path_suffix)
				if not os.path.exists(t1w_dst_file):
					shutil.copy(t1w_src_file, t1w_dst_file)
		else: 
			print(f'ERROR couldnt copy t1w of brain number {idx}')
		mask_src_file = os.path.join(src_dataset, idx, mask_src_path_suffix)
		mask_path_dst = os.path.join(dst_dataset, 'General-Masks', idx)
		if os.path.exists(mask_src_file) and os.path.isfile(mask_src_file):
			if not os.path.exists(mask_path_dst):
				os.mkdir(mask_path_dst)
			if os.path.isdir(mask_path_dst):
				mask_dst_file = os.path.join(mask_path_dst, mask_dst_path_suffix)
				if not os.path.exists(mask_dst_file):
					shutil.copy(mask_src_file, mask_dst_file)
		else:
			print(f'ERROR couldnt copy mask of brain number {idx}')
