# get a subsection for training
import torch
import numpy as np
from torch.nn import functional as F
import pandas as pd

image = np.array([10, 11, 12, 13, 14, 15])
image_length = image.shape[0]
x_tensor = torch.from_numpy(image)
print(x_tensor)
tensor_shape = x_tensor.shape


# Define a patch length, which will be the size of your extracted sub-section
patch_length = 3
# Define your start index
start_i = 2
# Define an end index given your start index and patch size
print(f"start index {start_i}")
end_i = start_i + patch_length
print(f"end index {end_i}")

# Extract a sub-section from your "image"
sub_section = image[start_i: end_i]
print("output patch length: ", len(sub_section))
print("output patch array: ", sub_section)

# Add one to your start index
start_i +=1

# Compute and print the largest valid value for start index
print(f"The largest start index for which "
      f"a sub section is still valid is "
      f"{image_length - patch_length}")

# Background Ratio
# We first simulate input data by defining a random patch of length 16. This will contain labels
# with the categories (0 to 3) as defined above.

patch_labels = np.random.randint(0, 3, (16))
y_tensor = torch.from_numpy(patch_labels)
print(patch_labels)
# A straightforward approach to get the background ratio is
# to count the number of 0's and divide by the patch length

bgrd_ratio = len(torch.where(y_tensor == 0)[0]) / len(y_tensor)
print("using torch.where(): ", bgrd_ratio)

# However, take note that we'll use our label array to train a neural network
# so we can opt to compute the ratio a bit later after we do some preprocessing.
# First, we convert the label's categories into one-hot format so it can be used to train the model
# patch_labels_one_hot = keras.utils.to_categorical(patch_labels, num_classes=4)
target = F.one_hot(y_tensor, num_classes=3)
print(target)

# Let's convert the output to a dataframe just so we can see the labels more clearly
pd.DataFrame(target, columns=['background', 'liver', 'tumor'])
print("background column: ", target[:, 0])
bgrd_ratio = torch.sum(target[:, 0]) / len(patch_labels)
print("using one-hot column: ", bgrd_ratio)


# z range differs considerably and varies between 42 and 1026
def get_sub_volume(image, label,
                   orig_x = 512, orig_y = 512, orig_z = 512,
                   output_x = 160, output_y = 160, output_z = 16,
                   num_classes = 4, max_tries = 1000,
                   background_threshold=0.95):
    """
        Extract random sub-volume from original images.

        Args:
            image (tensor): original image, of shape (orig_x, orig_y, orig_z, num_channels)
            label (tensor): original label. labels coded using discrete values rather than
                a separate dimension, so this is of shape (orig_x, orig_y, orig_z)
            orig_x (int): x_dim of input image
            orig_y (int): y_dim of input image
            orig_z (int): z_dim of input image
            output_x (int): desired x_dim of output
            output_y (int): desired y_dim of output
            output_z (int): desired z_dim of output
            num_classes (int): number of class labels
            max_tries (int): maximum trials to do when sampling
            background_threshold (float): limit on the fraction
                of the sample which can be the background

        returns:
            X (tensor): sample of original image of dimension
                (num_channels, output_x, output_y, output_z)
            y (tensor): labels which correspond to X, of dimension
                (num_classes, output_x, output_y, output_z)
        """
    to_padding = ([0, 0], [0, 0], [0, 0])

    if image.shape[2] < orig_z:
        to_padding[2][0] = (orig_z - image.shape[2]) // 2
        to_padding[2][1] = orig_z - image.shape[2] - to_padding[2][0]
        image = np.pad(image, to_padding, 'linear_ramp')
        label = np.pad(label, to_padding, 'linear_ramp')
    
    # Initialize features and labels with `None`
    X = None
    y = None

    tries = 0

    while tries < max_tries:
        # randomly sample sub-volume by sampling the corner voxel
        # hint: make sure to leave enough room for the output dimensions!
        start_x = np.random.randint(0, orig_x - output_x + 1)
        start_y = np.random.randint(0, orig_y - output_y + 1)
        start_z = np.random.randint(0, orig_z - output_z + 1)

