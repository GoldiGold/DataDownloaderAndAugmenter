import nibabel as nib
import os
import numpy as np
from nibabel.testing import data_path
from nilearn import plotting

np.set_printoptions(precision=2, suppress=True)

brain_id = 100307
mask_path = f'/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset/{brain_id}/T1w/1D Files/BA44_{brain_id}.1D.txt'
brain_path = f'/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset/{brain_id}/T1w/T1w_acpc_dc_restore_brain.nii.gz'
example_filename = os.path.join(data_path, 'example4d.nii.gz')
example_img = nib.load(example_filename)
img = nib.load(brain_path)
print(img.shape, img.get_data_dtype(), img.affine.shape, f'img 3D size is: 1st X 2nd X 3rd = {img.shape[0] * img.shape[1] * img.shape[2]}')
hdr = img.header
print(hdr.get_xyzt_units(), type(hdr), '\n', hdr.get_zooms())
# '''
with open(mask_path, 'r') as mask_file:
    mask = mask_file.read()
# print(type(mask), mask[:20])
mask = np.array([int(bit) for bit in mask if bit.isnumeric()])
print(np.sum(mask), 'the length of the mask is: ', len(mask))
exmp_hdr = example_img.header
print(f'the example mri scan header(sfrom_code)', exmp_hdr['sform_code'], '\n', exmp_hdr.get_sform(coded=True))

view2 = plotting.plot_anat(img)
print(type(view2))
# view2.open_in_browser()
# view22 = plotting.view_img(view2)
"""
view = plotting.view_img(img)
view.open_in_browser()
# THIS IS TO SHOW A SCAN IN THE BROWSER
"""
# view22.open_in_browser()
# vals = img.get_fdata()
# print(np.sum(np.where(vals > 0, 1, 0)))

# '''