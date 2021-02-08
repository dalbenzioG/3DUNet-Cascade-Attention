import numpy as np
import torch
from cascade_copy import *
from torch import optim, nn
from torch.utils.data import DataLoader
import torchsample.transforms as ts
import os
import SimpleITK as sitk

from image_loading import train_val_split, NiftiDataset, glob_imgs

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = cascade_UA_3D(in_dim=1).to(device)
batch_size = 1
learning_rate = 0.003
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

checkpoint = torch.load('/home/gabriella/gabri/results/3_Experiment/cascade_crossE+Tversky.ckpt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
tv_score = checkpoint['dice_score']

transform = ts.Compose([
    ts.ToTensor(),
    ts.ChannelsFirst(),
    ts.TypeCast(['float']),
    # ts.NormalizeMedicPercentile(norm_flag=True),
    ts.NormalizeMedic(norm_flag=True),
    ts.ChannelsLast(),
])

source_fns, target_fns = ["/home/gabriella/gabri/dataset/cropped_256/images/volume-54-resized.nii.gz"], \
                         ["/home/gabriella/gabri/dataset/cropped_256/labels/segmentation-54-resized.nii.gz"]
draw_map = NiftiDataset(source_fns=source_fns, target_fns=target_fns, transform=transform)

# Data loader

valid_loader = torch.utils.data.DataLoader(dataset=draw_map,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0
                                           )

model.eval()
with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        outputs = model(images)
        prediction_p = torch.log_softmax(outputs, dim=1)
        _, prediction = torch.max(prediction_p, dim=1)


output_arr = np.squeeze(prediction.cpu().byte().numpy()).astype(np.int16)
predi_img = sitk.GetImageFromArray(np.transpose(output_arr, (2, 1, 0)))
# predi_img = sitk.GetImageFromArray(output_arr)
# predi_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
predi_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
sitk.WriteImage(predi_img, os.path.join('/home/gabriella/gabri/dataset/stupid_map', 'pred_54.nii.gz'))
