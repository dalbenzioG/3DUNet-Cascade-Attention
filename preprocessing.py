import matplotlib
from scipy import ndimage
from scipy.ndimage import morphology

matplotlib.use('TkAgg')
import nibabel as nib
from nibabel import processing
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.ndimage.interpolation
from os.path import join

# # In Python, the glob module is used to retrieve files/pathnames matching a specified pattern.
root_dir = '/home/gabriella/gabri/dataset'
image_dir = join(root_dir, 'image/')
target_dir = join(root_dir, 'label/')
resample_label = join(root_dir, 'resample_label/')
resample_image = join(root_dir, 'resample_data/')
#image_filenames = sorted(glob.glob(image_dir + '**/*.nii.gz', recursive=True), key=lambda x: x[-16:])
target_filenames = sorted(glob.glob(target_dir + '/**/*.nii.gz', recursive=True), key=lambda x: x[-16:])

for images in target_filenames:
        nim = nib.load(images)
        raw_images = nim.get_fdata()  # return image data array
#       image_affine = nim.affine
#
        resampled_images = nib.processing.conform(nim, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0),
                                                  order=3, cval=0.0, orientation='RAS', out_class=None)

#     img = nib.Nifti1Image(resampled_images[0], np.eye(4))
#     img = nib.Nifti1Image(resize_images, np.eye(4))

        images_name = images.replace(target_dir, '')

        nib.save(resampled_images, resample_label + images_name[:-7] + '_resample.nii.gz')

