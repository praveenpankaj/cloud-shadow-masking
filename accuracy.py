"""
Accuracy of UNet model
 - Calculates from the patches generated with patch_gen script
 - Patch locations should be saved in csv file (cvs_paths script)
"""

from osgeo import gdal
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from sklearn.metrics import classification_report
from glob import glob
from sklearn.model_selection import train_test_split
from segmentation_models_pytorch import Unet
import torch.nn.functional as F


def calculate_iou_score(true_images, pred_images):
    assert len(true_images) == len(pred_images), "Number of true and predicted images must be the same."

    intersection_sum = 0
    union_sum = 0
    for i in range(len(true_images)):
        true_image = true_images[i]
        pred_image = pred_images[i]

        intersection = np.logical_and(true_image, pred_image)
        union = np.logical_or(true_image, pred_image)

        intersection_sum += np.sum(intersection)
        union_sum += np.sum(union)

    iou_score = intersection_sum / union_sum

    return iou_score


# create rasterised image from polygons:

data_path = '../data/cloud_data/patches2/'
image_paths = np.array(glob(data_path + "/images/*.tif"))
label_paths = np.array(glob(data_path + "/labels/*.tif"))
train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=42)
print(f'Train files: {len(train_idx)}, Valid files: {len(val_idx)}, Test files: {len(test_idx)}')
images_dir = image_paths[test_idx]
labels_dir = label_paths[test_idx]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model path
# PATH = '/home/mwv506/projects/roof_detection/ViT/trained_models4' \
#        '/best_resnet18_30p_103_31_05_23'
PATH = './trained_models3' \
       '/best_resnet152_30p_116_31_05_23'

n_channels = 13
num_classes = 3
# Model
model = Unet('resnet152 ', in_channels=n_channels, classes=num_classes)
model.to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
print("Model loaded sucessfully")

# Calculating accuracy
y, y_pred = [], []
intersection_sum, union_sum = 0, 0
for i in range(len(images_dir)):
    # Load the input images
    image_ = gdal.Open(images_dir[i])
    image_ = image_.ReadAsArray() / 10000
    image_ = torch.from_numpy(image_).float().to(device)
    image_ = image_.unsqueeze(0)
    with torch.no_grad():
        output = model(image_)
    output_array = F.softmax(output, dim=1).cpu().numpy()
    # save output class and score
    class_array = np.argmax(output_array, axis=1).squeeze()

    label = gdal.Open(labels_dir[i])
    label_array = label.ReadAsArray()-1
    y.append(label_array)
    y_pred.append(class_array)
    print("PredictBest_resnet34_30p_106_31_05_23ed " + str(i + 1) + " Patches")

iou_score = calculate_iou_score(y, y_pred)
print("\nIOU Score: ", iou_score)
# print classification report
y_true_, y_pred_ = np.array(y).flatten(), np.array(y_pred).flatten()
print(classification_report(y_true_, y_pred_, labels=list(range(num_classes))))

print("\n")
print("list of classes from predictions: " + str(np.unique(np.array(y_pred))))
print("list of classes from labels: " + str(np.unique(np.array(y))))
print("\n")
cm = confusion_matrix(np.array(y).flatten(), np.array(y_pred).flatten())

print("Confusion Matrix " + "\n")
print(cm, "\n")
accuracy = np.trace(cm/np.sum(cm))
print("Overal Accuracy: ", round(accuracy, 3), "\n")