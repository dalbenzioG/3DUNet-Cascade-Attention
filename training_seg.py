import torch

torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
from torch import optim, nn
from torch.utils.data import DataLoader
from loss import SoftDiceLoss, MultiTverskyLoss, tversky
from metrics import tv_index, dice_coefficient, dice_score

import matplotlib
import matplotlib.pyplot as plt

from cascade_copy import *
# from UnetCascade3D import *
from unet3D import *
from image_loading import train_val_split

import torchsample.transforms as ts

import os

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_STORE_PATH = '/home/gabriella/gabri/results/'

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 100
batch_size = 1
learning_rate = 0.003


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# For weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Data Augmentation
transform = ts.Compose([
    ts.ToTensor(),
    ts.ChannelsFirst(),
    ts.TypeCast(['float']),
    # ts.NormalizeMedicPercentile(norm_flag=True),
    ts.NormalizeMedic(norm_flag=True),
    ts.ChannelsLast(),
])

# Data split
train_dataset, valid_dataset = train_val_split("/home/gabriella/gabri/dataset/new/image",
                                               "/home/gabriella/gabri/dataset/new/label",
                                               valid_pct=0.2,
                                               transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0
                                           )

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0
                                           )

# Model, criterion and optimizer
model = cascade_UA_3D(in_dim=1).to(device)
# model = cascade_unet_3D(in_dim=1).to(device)
# model = Unet_3D(in_dim=1).to(device)
first_batch = next(iter(train_loader))
# if torch.cuda.current_device() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU
#     # model = nn.DataParallel(model)

print_network(model)
# criterion = SoftDiceLoss(n_classes=3)
# criterion = MultiTverskyLoss(n_classes=3)
# criterion = tversky(n_classes=3)
# weights = [0.3, 1.0, 1.0]
# class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

# betas=(0.9, 0.999), weight_decay=1e-5)
# model.parameter() refers to the weights adjustable by the optimizer
# learning rate = step's size to get the minimum of the loos function
# For updating learning rate
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.01, patience=10)
model.apply(weights_init)

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
print('Num training images: ', len(train_loader))
print('Num validation images: ', len(valid_loader))


log_interval = 20
dice_list = []
tv_list = []
loss_list = []
tv_val = []
dice_val =[]


step = len(first_batch)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)

    # for i, (images, labels) in enumerate([first_batch] * 50):
    for i, (images, labels) in enumerate(train_loader):
        # Move input and label tensors to the default device
        images = images.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, dim=1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.float(), labels.long())
        loss_list.append(loss.item())

        # Backward and optimize
        # all the optimizers implement a step() method, that updates the parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        dice_overall = dice_coefficient(labels.long(), outputs, num_classes=3) * 100
        tv_score = tv_index(labels.long(), outputs) * 100
        tv_list.append(tv_score.item())
        dice_list.append(dice_overall.item())

        # if (i + 1) % 100 == 0:
        if i % log_interval == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Tversky index (%): {:.4f}, Dice_overall: {:.4f}'.format(epoch + 1, num_epochs,
                                                                                                i + 1, total_step,
                                                                                                loss.item(),
                                                                                                tv_score.item(), dice_overall.item()))

        # Save the model checkpoint

        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'tv_score': tv_score.item(),

        }, MODEL_STORE_PATH + 'cascade_0702.ckpt')

# Test the model
# disables the gradient calculation.
model.eval()
with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, dim=1)
        outputs = model(images)

        dice_overall = dice_coefficient(labels.long(), outputs, num_classes=3) * 100
        tv_score = tv_index(labels.long(), outputs) * 100
        tv_val.append(tv_score.item())
        dice_val.append(dice_overall.item())
        print('Tversky Score: {:.4f}, Dice_overall: {:.4f}'.format(tv_score.item(), dice_overall.item()))

# # Training plot
x1 = range(1, len(loss_list) + 1)
x2 = range(1, len(tv_list) + 1)

y1 = loss_list
y2 = tv_list

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'darkorange')
plt.title('3DUnet training results (Cross Entropy)')
plt.ylabel('Cross Entropy')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'cornflowerblue')
plt.xlabel('Iterations')
plt.ylabel('Tversky coefficient')
plt.savefig("/home/gabriella/gabri/results/unet_train_crossE+Tversky.pdf")

# SCORE VALIDATION plotting
x_val = range(1, len(tv_val) + 1)
y_val = tv_val

fig, ax = plt.subplots()
ax.plot(x_val, y_val, 'cornflowerblue')
ax.set(xlabel='Number of samples', ylabel='Tversky coefficient',
       title='3DUnet validation result (Cross Entropy)')
ax.grid()
fig.savefig("/home/gabriella/gabri/results/unet_val_cross.pdf")

# second plot

# fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
# ax.plot(x_val, y_val, 'cornflowerblue')
# ax.set_xlabel(r'\textit{Number of samples}')
# ax.set_ylabel('\\textit{Dice coefficient}', fontsize=16)
# ax.set_title(r'\textbf{Cascade-Attention validation result (Dice)}', fontsize=16, color='k')
# fig.savefig("/home/gabriella/gabri/results/cascade_val_dice2.pdf")
