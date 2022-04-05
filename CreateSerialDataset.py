import os

serial_dataset_path = 'sdp'
og_dataset_path = 'odp'
og_t1w_path = 'otp'
og_masks_path = 'omp'
og_new_masks_path = 'omnp'

def create_serial_dataset(og_dataset: str, new_dataset: str):
	indices = [str(i) for i in os.listdir(src_dataset) if i.isdigit()]
	for sidx, idx in enumerate(indices):
		t1w_src_file = os.path.join(src_dataset, idx, t1w_src_path_suffix)
		t1w_path_dst = os.path.join(dst_dataset, 'T1w', sidx)
		if os.path.exists(t1w_src_file) and os.path.isfile(t1w_src_file):
			if os.path.isdir(t1w_path_dst):
				t1w_dst_file = os.path.join(t1w_path_dst, t1w_dst_path_suffix)
				if not os.path.exists(t1w_dst_file):
					copy(t1w_src_file, t1w_dst_file)
		else: 
			print(f'ERROR couldnt copy t1w of brain number {idx}')
		mask_src_file = os.path.join(src_dataset, idx, mask_src_path_suffix)
		mask_path_dst = os.path.join(dst_dataset, 'General-Masks', sidx)
		if os.path.exists(mask_src_file) and os.path.isfile(mask_src_file):
			if not os.path.exists(mask_path_dst):
				os.mkdir(mask_path_dst)
			if os.path.isdir(mask_path_dst):
				mask_dst_file = os.path.join(mask_path_dst, mask_dst_path_suffix)
				if not os.path.exists(mask_dst_file):
					copy(mask_src_file, mask_dst_file)
		else:
			print(f'ERROR couldnt copy mask of brain number {idx}')
	
