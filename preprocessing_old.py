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


data_path = '/home/user1/Downloads/Training Batch 1/'
#labelname = 'segmentation-30.nii.gz'
imagename = 'volume-0.nii.gz'
#file_path1 = join(data_path, labelname)
file_path = join(data_path, imagename)
medical_image = nib.load(file_path)
print(medical_image.affine)
#medical_image1 = nib.load(file_path1)
image = medical_image.get_fdata()
#image1 = medical_image1.get_fdata()
# spacing_voxel = np.array(list(medical_image.header.get_zooms()))
# image_dim_image = medical_image.header.get_data_shape()
# image_dim_label = medical_image1.header.get_data_shape()
#
#
# We have to transform the pixel values to the Hounsfield units. We can achieve this using the headers in the
# medical_image file, we will use the Rescale Intercept and Rescale Slope headers
def transform_to_hu(medical_image, image):
    intercept = medical_image.dataobj.inter
    slope = medical_image.dataobj.slope
    hu_image = image * slope + intercept

    return hu_image

#
# def sample_stack(file_path, rows=6, cols=6, start_with=200, show_every=10):
#     medical_image = nib.load(file_path)
#     image = medical_image.get_fdata()
#     hu_image = transform_to_hu(medical_image, image)
#     fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
#     for i in range(rows * cols):
#         ind = start_with + i * show_every
#         slice_img = hu_image[:, :, ind]
#         ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
#         ax[int(i / rows), int(i % rows)].imshow(np.rot90(slice_img), cmap='gray')
#         ax[int(i / rows), int(i % rows)].axis('off')
#     plt.show()


# width = 160, window center = 60
# If we want a specific zone of the image we can windowing the image
def window_image(medical_image, image, img_min, img_max):
    # img_min = window_center - window_width // 2
    # img_max = window_center + window_width // 2
    img = transform_to_hu(medical_image, image)
    window_image = img.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image


def resample(medical_image, image, new_spacing=[1, 1, 1]):
    """
    medical_image = nib.load(file_path)
    image = medical_image.get_fdata()
    """
    image_shape = np.array(list(medical_image.header.get_data_shape()))
    spacing = np.array(list(medical_image.header.get_zooms()))
    resize_factor = spacing / new_spacing

    new_shape = image_shape * resize_factor
    new_shape = np.round(new_shape)

    real_resize_factor = new_shape / image_shape

    new_spacing = spacing / real_resize_factor
    resampled_image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return resampled_image, new_spacing


# def display_views(file_path):
#     image_axis = 2
#     medical_image = nib.load(file_path)
#     image = medical_image.get_fdata()
#     hu_image = transform_to_hu(medical_image, image)
#     liver_image = window_image(medical_image, image, -1000, 400)
#
#     sagital_image = image[300, :, :]  # Axis 0
#     coronal_image = image[:, 300, :]  # Axis 1
#     axial_image = image[:, :, 60]  # Axis 2
#     axial_image_w = liver_image[:, :, 60]
#
#     plt.figure(figsize=(20, 10))
#     plt.style.use('grayscale')
#
#     plt.subplot(241)
#     plt.imshow(np.rot90(sagital_image))
#     plt.title('Sagital Plane')
#     plt.axis('off')
#
#     plt.subplot(242)
#     plt.imshow(np.rot90(axial_image))
#     plt.title('Axial Plane')
#     plt.axis('off')
#
#     plt.subplot(243)
#     plt.imshow(np.rot90(coronal_image))
#     plt.title('Coronal Plane')
#     plt.axis('off')
#
#     plt.subplot(244)
#     plt.hist(hu_image.flatten(), bins=50, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency_before windowing")
#
#     plt.subplot(245)
#     plt.hist(liver_image.flatten(), bins=50, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency_after windowing")
#
#     plt.subplot(246)
#     plt.imshow(np.rot90(axial_image_w))
#     plt.title('Axial Plane_after windowing')
#     plt.axis('off')


def resize_data(initial_data):
    initial_size_x = initial_data.shape[0]
    initial_size_y = initial_data.shape[1]
    initial_size_z = initial_data.shape[2]

    new_size_x = 480
    new_size_y = 480
    new_size_z = 480

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x in range(new_size_x):
        for y in range(new_size_y):
            for z in range(new_size_z):
                new_data[x][y][z] = initial_data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data


def remove_noise(file_path, display=False):
    medical_image = nib.load(file_path)
    image = medical_image.get_data()
    liver_window = window_image(image, 60, 160)

    # morphology.dilation creates a segmentation of the image
    # If one pixel is between the origin and the edge of a square of size
    # 5x5, the pixel belongs to the same class

    # We can instead use a circule using: morphology.disk(2)
    # In this case the pixel belongs to the same class if it's between the origin
    # and the radius

    segmentation = morphology.dilation(liver_window, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int))
    # The size of label_count is the number of classes/segmentations found

    # We don't use the first class since it's the background
    label_count[0] = 0

    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()

    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    # Since the the pixels in the mask are zero's and one's
    # We can multiple the original image to only keep the brain region
    masked_image = mask * liver_window

    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(liver_window)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(masked_image)
        plt.title('Final Image')
        plt.axis('off')

    return masked_image


# # In Python, the glob module is used to retrieve files/pathnames matching a specified pattern.
root_dir = '/home/user1/gabri/dataset/'
image_dir = join(root_dir, 'image/')
target_dir = join(root_dir, 'label/')
resample_label = join(root_dir, 'resample_label/')
resample_image = join(root_dir, 'resample_data/')
image_filenames = sorted(glob.glob(image_dir + '**/*.nii.gz', recursive=True), key=lambda x: x[-18:])
# # target_filenames = sorted(glob.glob(tar_dir + '/**/*.nii.gz', recursive=True))

for images in image_filenames:
    nim = nib.load(images)
    raw_images = nim.get_fdata()  # return image data array
    image_affine = nim.affine

    #resampled_images = resample(nim, raw_images, new_spacing=[1, 1, 1])
    resize_images = resize_data(raw_images)
    #img = nib.Nifti1Image(resampled_images[0], np.eye(4))
    img = nib.Nifti1Image(resize_images, np.eye(4))
    images_name = images.replace(image_dir, '')
    # images_name = images.replace(image_dir, '')

#     nib.save(img, resample_dir + images_name[:-7] + '_resample.nii.gz')

resize_images = resize_data(image)
img = nib.Nifti1Image(resize_images, np.eye(4))
