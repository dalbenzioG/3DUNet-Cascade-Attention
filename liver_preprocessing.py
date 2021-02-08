from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib


#matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from nibabel import processing
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.ndimage.interpolation
from os.path import join
import math


# # In Python, the glob module is used to retrieve files/pathnames matching a specified pattern.
root_dir = '/home/gabriella/gabri/dataset'
image_dir = join(root_dir, 'image/')
target_dir = join(root_dir, 'label/')
resample_label = join(root_dir, 'resample_label/')
resample_image = join(root_dir, 'resample_data/')
#image_filenames = sorted(glob.glob(image_dir + '**/*.nii.gz', recursive=True), key=lambda x: x[-16:])
target_filenames = sorted(glob.glob(target_dir + '/**/*.nii.gz', recursive=True), key=lambda x: x[-16:])


# ct_path='D:/Science/Github/3D-Medical-Imaging-Preprocessing-All-you-need/Data/img0001.nii.gz'
# ct_label_path='D:/Science/Github/3D-Medical-Imaging-Preprocessing-All-you-need/Data/label0001.nii.gz'

# # CT
# img_sitk  = sitk.ReadImage(ct_path, sitk.sitkFloat32) # Reading CT
# image     = sitk.GetArrayFromImage(img_sitk) #Converting sitk_metadata to image Array
# # Mask
# mask_sitk = sitk.ReadImage(ct_label_path,sitk.sitkInt32) # Reading CT
# mask      = sitk.GetArrayFromImage(mask_sitk)#Converting sitk_metadata to image Array
#
# print('CT Shape={}'.format(image.shape))
# print('CT Mask Shape={}'.format(mask.shape))


def normalise(image):
    # normalise and clip images -1000 to 800
    np_img = image
    np_img = np.clip(np_img, -2000., 800.).astype(np.float32)
    return np_img

def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)


# Crop to [64, 64, 64] smooth-padding
#img_cropped = resize_image_with_crop_or_pad(image, [96, 96, 96], mode='symmetric')

# for images in target_filenames:
#         nim = nib.load(images)
#         raw_images = nim.get_fdata()  # return image data array
# #       image_affine = nim.affine
# #
#         resampled_images = nib.processing.conform(nim, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0),
#                                                   order=3, cval=0.0, orientation='RAS', out_class=None)
#
# #     img = nib.Nifti1Image(resampled_images[0], np.eye(4))
# #     img = nib.Nifti1Image(resize_images, np.eye(4))
#
#         images_name = images.replace(target_dir, '')
#
#         nib.save(resampled_images, resample_label + images_name[:-7] + '_resample.nii.gz')

image = join(root_dir,image_dir, 'volume-7.nii.gz')
nim = nib.load(image)
array_image = nim.get_fdata()

image_cropped = resize_image_with_crop_or_pad(array_image, [256, 256, 256], mode='symmetric')

# Visualise using matplotlib.
f, axarr = plt.subplots(1, 2, figsize=(15,15))
axarr[0].imshow(np.squeeze(array_image[100, :, :]), cmap='gray',origin='lower')
axarr[0].axis('off')
axarr[0].set_title('Original image {}'.format(array_image.shape))

axarr[1].imshow(np.squeeze(image_cropped[100, :, :]), cmap='gray',origin='lower')
axarr[1].axis('off')
axarr[1].set_title('Cropped to {}'.format(image_cropped.shape))

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)