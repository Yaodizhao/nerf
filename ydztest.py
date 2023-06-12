import io

import cv2
import numpy as np
from osgeo import gdal
from PIL import Image


gdal.UseExceptions()
path_image = "/home/dev08/test1.tif"
source_image = "/home/dev08/MS600_0004_450nm.tif"

ds_features = gdal.Open(path_image, 1)

image_open = Image.open(path_image)


# image_open.save("/home/dev08/test1.tif")
band1 = ds_features.GetRasterBand(1)
band1 = band1.ReadAsArray()
print(band1.shape)

