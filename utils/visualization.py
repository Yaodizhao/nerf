import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    将深度权重 变成0-255的矩阵并变成灰度rgb图像
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # 把NaN值赋成0
    mi = np.min(x)  # 获得最大和最小深度
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # 正则化到 0~1
    x = (255 * x).astype(np.uint8)
    # 对于0~255范围的灰度值映射成JET模式的伪彩
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_
