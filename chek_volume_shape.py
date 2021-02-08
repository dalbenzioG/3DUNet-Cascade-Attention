import nibabel as nib

import glob
from os.path import join

# # In Python, the glob module is used to retrieve files/pathnames matching a specified pattern.
root_dir = '/home/gabriella/gabri/dataset/'
image_dir = join(root_dir, 'images/')
target_dir = join(root_dir, 'labels/')


image_filenames = sorted(glob.glob(image_dir + '**/*.nii.gz', recursive=True), key=lambda x: x[-17:])

target_filenames = sorted(glob.glob(target_dir + '/**/*.nii.gz', recursive=True), key=lambda x: x[-17:])
print("\n".join(target_filenames))
for images in target_filenames:
        nim = nib.load(images)
        raw_images = nim.get_fdata()  # return image data array
        print('image_shape {}'.format(raw_images.shape))
