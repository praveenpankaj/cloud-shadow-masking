import glob
import os.path
import numpy as np
import torch
from segmentation_models_pytorch import Unet
import torch.nn.functional as F
from tqdm import tqdm
from osgeo import gdal
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gdal_raster_utils import write_image
# Model path
PATH = '/home/mwv506/projects/roof_detection/ViT/trained_models3' \
       '/best_resunet_30p_81_31_05_23'

# test images
test = False
valid = True
train = False
outfile_name = "resunet_model"
save_images = True

# load model
n_channels = 13
num_classes = 3
model = Unet('resnet18', in_channels=n_channels, classes=num_classes)
model.cuda()
model.load_state_dict(torch.load(PATH))
model.eval()
print("Model loaded sucessfully")

images = ['../data/cloud_data/images/S2A_MSIL1C_20220101T052231_N0301_R062_T43QEU_20220101T072023.tif']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for image in tqdm(images):
    # Load the input images
    _image = gdal.Open(image)
    image_data_type = _image.GetRasterBand(1).DataType

    image_ = _image.ReadAsArray()/10000

    image_ = image_[:, 4500:4756, 4500:4756]
    tes_img = image_.transpose(1,2,0)*10000
    image_ = torch.from_numpy(image_).float().to(device)
    # _image = torch.movedim(_image, -1, 0)
    image_ = image_.unsqueeze(0)
    with torch.no_grad():
        output = model(image_)
    output_array = F.softmax(output, dim=1).cpu().numpy()

    # save output class and score
    class_array = np.argmax(output_array, axis=1).squeeze()

    # convert results to json
    fname = "test3_" + os.path.basename(image)

    write_image(tes_img, fname, projection=_image.GetProjection(), transform=_image.GetGeoTransform(),
                data_type=image_data_type)
