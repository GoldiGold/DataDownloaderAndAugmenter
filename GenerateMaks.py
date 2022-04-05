import os
import json

def generate_masks(t1w_path: str, new_masks_path: str, mode: str, json_file_path: str):
	indices = [str(i) for i in os.listdir(t1w_path) if i.isdigit()] #if this is the correct syntax
	if mode == 'first':
		# read json file label 2 new label	
		# read json file label 2 value	
		subdir = '1st subdir'		
		mask_name = '1st mask'
		continue
	elif mode == 'second':
		# read json file label 2 new label	
		# read json file label 2 value			
		subdir = '2nd subdir'
		mask_name = '2nd mask'
		continue
	elif mode == 'third':
		# read json file label 2 new label	
		# read json file label 2 value		
		subdir = '3rd subdir'	
		mask_name = '3rd mask'
		continue
	elif mode == 'fourth':
		# read json file label 2 new label	
		# read json file label 2 value	
		subdir = '4th subdir'		
		mask_name = '4th mask'
		continue
	masks_counter = 0
	existed_counter = 0
	for idx in indices:
		mask_dir = os.path.join(new_masks_path, subdir, idx)
		if !os.path.exists(mask_dir):
			os.mkdir(mask_dir)
		if !os.path.exists(os.path.join(mask_dir, mask_name)):
			#enter mask creation function from our code with label -> new label -> value
			masks_counter += 1
		else:
			existed_counter += 1
	return masks_counter, existed_counter 
					
