import torch
from torch.utils import tensorboard
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from osgeo import gdal
from torchmetrics import Accuracy, JaccardIndex
from segmentation_models_pytorch import Unet
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from PIL import Image

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print("Cuda is available...")


class TIFDataset(Dataset):
    def __init__(self, images, labels, transform=None, sentinel_data=True):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        # import ipdb
        # ipdb.set_trace()
        image = gdal.Open(image_path)
        # rgb = image.ReadAsArray()[1:4, :, :] / 10000  #
        # nir = image.ReadAsArray()[7:8, :, :] / 10000
        # image = np.concatenate((rgb, nir), axis=0)
        image = image.ReadAsArray() / 10000
        image = torch.from_numpy(image).float().to(device)
        label = gdal.Open(label_path).ReadAsArray()
        label = torch.from_numpy(label).long().to(device)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# Define the hyperparameters
num_classes = 2
batch_size = 16
epochs = 200
learning_rate = 0.001
outmodel_dir = './trained_models4/'
model_name = 'resnet18'
keyword = '30p_4c'
n_channels = 13

# Training and validation paths
# Training paths
# images_dir = '/home/venky/Documents/diku/projects/roof_detection/sample_data/roof_type_bm/unet/image/'
# labels_dir = '/home/venky/Documents/diku/projects/roof_detection/sample_data/roof_type_bm/unet/roof_type_class/'
# Validation paths
# val_images_dir = '/home/venky/Documents/diku/projects/roof_detection/sample_data/roof_type_bm/unet/image/'
# val_labels_dir = '/home/venky/Documents/diku/projects/roof_detection/sample_data/roof_type_bm/unet/roof_type_class/'

data_path = '../data/cloud_data/patches3/'
image_paths = np.array(glob.glob(data_path + "/images/*.tif"))
label_paths = np.array(glob.glob(data_path + "/labels/*.tif"))
train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=42)
print(f'Train files: {len(train_idx)}, Valid files: {len(val_idx)}, Test files: {len(test_idx)}')
images_dir = image_paths[train_idx]
labels_dir = label_paths[train_idx]
val_images_dir = image_paths[val_idx]
val_labels_dir = label_paths[val_idx]

# Create dataloaders
# Training
dataset = TIFDataset(images_dir, labels_dir)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Validation
val_dataset = TIFDataset(val_images_dir, val_labels_dir)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Model
model = Unet(model_name, in_channels=n_channels, classes=num_classes, encoder_weights='imagenet')

# training
writer = tensorboard.SummaryWriter(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# iou_m = JaccardIndex(task="multiclass", num_classes=num_classes, absent_score=0).to(device)
iou_m = JaccardIndex(task="binary", absent_score=0).to(device)
accuracy_m = Accuracy(task="multiclass", top_k=1, num_classes=num_classes).to(device)
best_iou = -1

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    loss_avg = 0
    pbar_train = tqdm(enumerate(train_loader), desc="unet")
    for i, (X, y) in pbar_train:
        pred = model(X)
        loss = F.cross_entropy(pred, y.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss = criterion(outputs, targets)
        loss_avg += loss.item()
        pbar_train.set_postfix_str(f"loss: {loss_avg / (i + 1)}")

    pbar_train.reset()
    # log last images
    # print("Test here----------------", y.shape, y[:1, :, :].shape)
    writer.add_images("unet/input_rgb", X[:, 1:4], global_step=epoch)  # change to RGB
    writer.add_images("unet/label", y[:1, :, :] / num_classes, global_step=epoch, dataformats="CHW")
    writer.add_images("unet/pred", pred.argmax(1, keepdims=True)[0] / num_classes, global_step=epoch, dataformats="CHW")
    writer.add_scalar("unet/loss", loss_avg / len(train_loader), global_step=epoch)
    model.eval()
    loss_avg = 0
    iou_m.reset()
    accuracy_m.reset()

    pbar_val = tqdm(enumerate(val_loader), desc="val")
    for i, (X, y) in pbar_val:
        with torch.no_grad():
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            accuracy_m.update(pred, y.squeeze(1))
            iou_m.update(pred, y.squeeze(1))

            loss = F.cross_entropy(pred, y.squeeze(1))
            loss_avg += loss.item()
            pbar_val.set_postfix_str(f"loss: {loss_avg / (i + 1)}")
    pbar_val.reset()


    writer.add_images("val/input_rgb", X[:, 1:4], global_step=epoch)
    writer.add_images("val/label", y[:1, :, :] / num_classes, global_step=epoch,
                      dataformats="CHW")
    writer.add_images("val/pred", pred.argmax(1, keepdims=True)[0] / num_classes, global_step=epoch, dataformats="CHW")

    writer.add_scalar("val/loss", loss_avg / len(train_loader), global_step=epoch)
    writer.add_scalar("val/acc", accuracy_m.compute(), global_step=epoch)
    iou = iou_m.compute()
    writer.add_scalar("val/iou", iou, global_step=epoch)

    # save model every 5 epochs
    if epoch % 5 == 0:
        torch.save(
            model.state_dict(),
            outmodel_dir + f"latest{model_name}_{keyword}_{epoch}_{datetime.today().strftime('%d_%m_%y')}"
        )

    if best_iou < iou:
        best_iou = iou
        torch.save(
            model.state_dict(),
            outmodel_dir + f"best_{model_name}_{keyword}_{epoch}_{datetime.today().strftime('%d_%m_%y')}"
        )
