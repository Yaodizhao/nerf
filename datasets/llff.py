import glob
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import *


def normalize(v):
    """ 规范化向量 """
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    计算平均姿势，然后用于居中所有姿势
    使用@center_poses。其计算如下：

    1. 计算中心：姿势中心的平均值。

    2. 计算 z 轴：归一化平均 z 轴。

    3. 计算轴 y'：平均 y 轴。

    4. 计算 x' = y' 叉积 z，然后将其规范化为 x 轴。

    5. 计算 y 轴：z 交叉乘积 x。

    注意，在步骤 3 中，我们不能直接使用 y' 作为 y 轴，
    因为它不一定与 z 轴正交。我们需要从 x 传递到 y。

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. 计算中心
    center = poses[..., 3].mean(0)  # (3)

    # 2. 计算z轴
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. 计算轴 y'(无需归一化，因为它不是最终输出)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. 计算x轴
    x = normalize(np.cross(y_, z))  # (3)

    # 5. 计算y轴(当z和x归一化时，y已经是归一化后的范数1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    将姿势居中，以便我们可以使用 NDC。
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) 居中姿势
        pose_avg: (3, 4) 平均姿势
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)  # 4*4的矩阵对角线值为1
    pose_avg_homo[:3] = pose_avg  # 转换为齐次坐标以加快计算速度
    # 在每个pose最后一行添加 0, 0, 0, 1
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) 齐次坐标

    # 使用平均位姿的逆矩阵 左乘 相机位姿 使得 平均相机位姿与世界坐标系一致
    pose_avg_homo_inv = np.linalg.inv(pose_avg_homo)
    poses_centered = pose_avg_homo_inv @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo_inv


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378), spheric_poses=False, val_num=1):
        """
        spheric_poses: 图像是否以球形向内方式拍摄 默认值：flase（前向）
        val_num: 验证图像数（用于多 GPU 训练，验证所有 GPU 的相同图像
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num)  # 至少有一个GPU

        # 定义转换方法
        # 对于形状：H x W x C -> C x H x W
        # 对于数值：[0， 255] 归一化到 [0.0， 1.0] 范围内
        self.define_transforms()

        self.read_meta()

        # 由于不是合成图片白色背景设为false
        self.white_back = False

    def read_meta(self):
        # (N_images, 17)
        # r11~r33 c2w旋转矩阵
        # t1~t3 c2w平移向量
        # H 照片高度 W 照片宽度 f 相机焦距
        # [r11 r12 r13 t1 H r21 r22 r23 t2 W r31 r32 r33 t3 f near far]
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy'))

        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))

        # 加载全分辨率图像，然后调整大小
        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                '图像数量和姿势数量不匹配！请重新运行 COLMAP！'

        # 转换成矩阵
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        # 定义far和near点
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # 第1步：根据训练分辨率重新缩放焦距
        # 原始相机内参，所有图像共用一个  H, W, focal
        H, W, self.focal = poses[0, :, -1]
        assert H * self.img_wh[0] == W * self.img_wh[1], \
            f'必须将@img_wh设置为与 ({W}， {H})具有相同的纵横比 !'

        # 焦距也要除以因数？
        self.focal *= self.img_wh[0] / W

        # 第2步：校正位姿
        # 原始姿势以“下 右 后”的形式旋转，更改为“右 上 后”
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) 排除 H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)  # 根据pose的平移向量求距离
        val_idx = np.argmin(distances_from_center)  # 选择最接近中心的图像获取索引

        # 第3步: 正确缩放边界以使位置编码有意义, 使最近的深度略大于1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75是默认参数
        # 最近的深度是 1/0.75=1.33 使用bound除以scale因数
        self.bounds /= scale_factor
        # 把平移向量除scale因数
        self.poses[..., 3] /= scale_factor

        # 所有像素的光线方向，所有图像的光线方向相同（相同的H，W，焦距）
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)  # (H, W, 3)

        if self.split == 'train':  # 创建所有光线和 RGB 数据的缓冲区
            # 使用第一个 N_images-1 进行训练，最后一个是 val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx:  # 排除 val 图像
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1] * self.img_wh[0] == img.size[0] * self.img_wh[1], \
                    f'''{image_path} 宽高比与img_wh不同，请检查您的数据!'''

                # 重采样数据
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                # 使用ndc坐标系中将光线的边界限制在0-1之间以减少在景深比较大的背景下产生的负面影响
                # 注意 ndc坐标系不能和 spherify pose（球状多视图采样时）同时使用
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                    # near plane is always at 1.0
                    # near and far in NDC are always 0 and 1
                    # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max())  # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             near * torch.ones_like(rays_o[:, :1]),
                                             far * torch.ones_like(rays_o[:, :1])],
                                            1)]  # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0)  # ((N-1) * H * W, 8)  排除最接近中心的图像
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N-1) * H * W, 3)

        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]

        else:  # 为了进行测试，请创建参数化渲染路径
            if self.split.endswith('train'):  # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5  # hardcoded, this is numerically close to the formula
                # given in the original repo. Mathematically if near=1
                # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train':  # 使用缓冲区中的数据
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d,
                              near * torch.ones_like(rays_o[:, :1]),
                              far * torch.ones_like(rays_o[:, :1])],
                             1)  # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
                sample['rgbs'] = img

        return sample
