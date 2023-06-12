import torch
from kornia import create_meshgrid


def get_ray_directions(H, W, focal):
    """
    获取相机坐标中所有像素的光线方向
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), 相机坐标中光线的方向
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]  # (H, W, 2) 像素坐标
    # i: (H, W) (0, 1, 2, ... W) * H
    # j: (H, W)  W个行数的值 * H
    i, j = grid.unbind(-1)
    # 此处的方向没有 +0.5 像素居中，因为校准不是那么准确
    # ((i-cx)/f, (j-cy)/f, -1)
    # 将相机坐标转换为opengl坐标(x right y up z back)因此 -y, z = -1
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    获取一个图像中所有像素的世界坐标中的光线原点和归一化方向。
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) 相机坐标中预先计算的光线方向
        c2w: (3, 4) 从相机坐标到世界坐标的变换矩阵

    Outputs:
        rays_o: (H*W, 3), 光线在世界坐标中的原点
        rays_d: (H*W, 3), 世界坐标中光线的归一化方向
    """
    # 将光线方向从相机坐标旋转到世界坐标
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    # 归一化把值限制在(-1,1)之间
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # 所有光线的原点是世界坐标中的相机原点
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    # 变成(H*W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    将光线从世界坐标转换为 NDC。NDC：空间，使画布是一个立方体，每个轴上都有边 [-1， 1]。
    For detailed derivation, please see:
    参考 /ndc_helpers/ndc_derivation.pdf
    https://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    在实践中，使用NDC“当且仅当”场景是无界的（具有较大的深度）
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: 图像高度，宽度和焦距
        near: (N_rays) or float, 近平面的深度
        rays_o: (N_rays, 3), 光线在世界坐标中的原点
        rays_d: (N_rays, 3), 光线在世界坐标中的方向

    Outputs:
        rays_o: (N_rays, 3), NDC中射线的起源
        rays_d: (N_rays, 3), NDC中射线的方向
    """
    # 将射线原点移动到近平面 rays_o[3]全变成近平面 1
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # 存储一些中间均匀的结果
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # 投影
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (H*W, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (H*W, 3)

    return rays_o, rays_d
