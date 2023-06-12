from argparse import ArgumentParser
from collections import defaultdict

import cv2
import mcubes
import numpy as np
import open3d as o3d
from PIL import Image
from PIL.Image import Resampling
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from datasets import dataset_dict
from models.nerf import *
from models.rendering import *
from utils import load_ckpt

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output ply filename')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of samples to infer the acculmulated opacity')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--N_grid', type=int, default=256,
                        help='size of the grid on 1 side, larger=higher resolution')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--sigma_threshold', type=float, default=20.0,
                        help='threshold to consider a location is occupied')
    parser.add_argument('--occ_threshold', type=float, default=0.2,
                        help='''threshold to consider a vertex is occluded.
                                larger=fewer occluded pixels''')

    # method using vertex normals #
    parser.add_argument('--use_vertex_normal', action="store_true",
                        help='use vertex normals to compute color')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of fine samples to infer the acculmulated opacity')
    parser.add_argument('--near_t', type=float, default=1.0,
                        help='the near bound factor to start the ray')

    return parser.parse_args()


@torch.no_grad()
def f(models, embeddings, rays, N_samples, N_importance, chunk, white_back):
    """使用块对光线进行批量推理."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i + chunk],
                        N_samples,
                        False,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()

    kwargs = {'root_dir': args.root_dir,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = True
        kwargs['split'] = 'test'
    else:
        kwargs['split'] = 'train'
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    embeddings = [embedding_xyz, embedding_dir]
    nerf_fine = NeRF()
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda().eval()

    # define the dense grid for query
    N = args.N_grid
    xmin, xmax = args.x_range
    ymin, ymax = args.y_range
    zmin, zmax = args.z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    # sigma 与方向无关，因此此处的任何值都将产生相同的结果

    # 预测每个网格位置的sigma（透明度）
    print('Predicting occupancy ...')
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, args.chunk)):
            xyz_embedded = embedding_xyz(xyz_[i:i + args.chunk])  # (N, embed_xyz_channels)
            dir_embedded = embedding_dir(dir_[i:i + args.chunk])  # (N, embed_dir_channels)
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
            out_chunks += [nerf_fine(xyzdir_embedded)]
        rgbsigma = torch.cat(out_chunks, 0)

    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0).reshape(N, N, N)

    # 执行行进立方体算法以检索顶点和三角形网格
    print('Extracting mesh ...')
    vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)

    # 直到此处的网格提取，它与原始存储库相同。

    vertices_ = (vertices / N).astype(np.float32)
    # 反转 x 和 y 坐标（为什么？也许是因为行进立方体算法）
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'),
             PlyElement.describe(face, 'face')]).write(f"{args.scene_name}.ply")

    # remove noise in the mesh by keeping only the biggest cluster
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(f"{args.scene_name}.ply")
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices) / 1e6:.2f} M vertices and {len(mesh.triangles) / 1e6:.2f} M faces.')

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

    # 执行颜色预测
    # Step 0. 定义常量（图像宽度、高度和内部函数）
    W, H = args.img_wh
    K = np.array([[dataset.focal, 0, W / 2],
                  [0, dataset.focal, H / 2],
                  [0, 0, 1]]).astype(np.float32)

    # Step 1. 将顶点转换为世界坐标
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1)  # (N, 4)

    if args.use_vertex_normal:  # 按照作者的建议使用法向量法
        # see https://github.com/bmild/nerf/issues/44
        mesh.compute_vertex_normals()
        rays_d = torch.FloatTensor(np.asarray(mesh.vertex_normals))
        near = dataset.bounds.min() * torch.ones_like(rays_d[:, :1])
        far = dataset.bounds.max() * torch.ones_like(rays_d[:, :1])
        rays_o = torch.FloatTensor(vertices_) - rays_d * near * args.near_t

        nerf_coarse = NeRF()
        load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
        nerf_coarse.cuda().eval()

        results = f([nerf_coarse, nerf_fine], embeddings,
                    torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                    args.N_samples,
                    args.N_importance,
                    args.chunk,
                    dataset.white_back)

    else:  # 使用颜色平均方法。见README_mesh.md
        # 用于存储最终平均颜色的缓冲区
        non_occluded_sum = np.zeros((N_vertices, 1))
        v_color_sum = np.zeros((N_vertices, 3))

        # Step 2. 将顶点投影到每个训练图像上以推断颜色
        print('Fusing colors ...')
        for idx in tqdm(range(len(dataset.image_paths))):
            # 阅读此姿势的图像
            image = Image.open(dataset.image_paths[idx]).convert('RGB')
            # image = image.resize(tuple(args.img_wh), Image.LANCZOS)
            image = image.resize(tuple(args.img_wh), Resampling.LANCZOS)
            image = np.array(image)

            # 读取相机到世界相对姿势
            P_c2w = np.concatenate([dataset.poses[idx], np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            P_w2c = np.linalg.inv(P_c2w)[:3]  # (3, 4)
            # 将顶点从世界坐标投影到照相机坐标
            vertices_cam = (P_w2c @ vertices_homo.T)  # (3, N) in "right up back"
            vertices_cam[1:] *= -1  # (3, N) in "right down forward"
            # 将顶点从照相机坐标投影到像素坐标
            vertices_image = (K @ vertices_cam).T  # (N, 3)
            depth = vertices_image[:, -1:] + 1e-5  # 顶点的深度，用作远平面
            vertices_image = vertices_image[:, :2] / depth
            vertices_image = vertices_image.astype(np.float32)
            vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W - 1)
            vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H - 1)

            # 使用双线性插值计算这些投影像素坐标上的颜色。注意：opencv 的实现具有每边 32768 像素的大小限制，因此我们将输入分成块。
            colors = []
            remap_chunk = int(3e4)
            for i in range(0, N_vertices, remap_chunk):
                colors += [cv2.remap(image,
                                     vertices_image[i:i + remap_chunk, 0],
                                     vertices_image[i:i + remap_chunk, 1],
                                     interpolation=cv2.INTER_LINEAR)[:, 0]]
            colors = np.vstack(colors)  # (N_vertices, 3)

            # 预测每个顶点的遮挡 我们利用NeRF的概念，构建从相机出来并击中每个顶点的光线;
            # 通过计算沿此路径的累积不透明度，我们可以知道顶点是否被遮挡。
            # 对于似乎被每个输入视图遮挡的顶点，我们假设它的颜色与面向我们一侧的邻居相同。
            # （想象一个一侧面向我们的表面：我们假设另一侧具有相同的颜色）

            # 光线的起源是相机的起源
            rays_o = torch.FloatTensor(dataset.poses[idx][:, -1]).expand(N_vertices, 3)
            # 光线的方向是从相机原点指向顶点的矢量
            rays_d = torch.FloatTensor(vertices_) - rays_o  # (N_vertices, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            near = dataset.bounds.min() * torch.ones_like(rays_o[:, :1])
            # 远平面是顶点的深度，因为我们想要的是从相机原点到顶点的路径上的累积不透明度
            far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
            results = f([nerf_fine], embeddings,
                        torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                        args.N_samples,
                        0,
                        args.chunk,
                        dataset.white_back)
            opacity = results['opacity_coarse'].cpu().numpy()[:, np.newaxis]  # (N_vertices, 1)
            opacity = np.nan_to_num(opacity, 1)

            non_occluded = np.ones_like(non_occluded_sum) * 0.1 / depth  # weight by inverse depth
            # near=more confident in color
            non_occluded += opacity < args.occ_threshold

            v_color_sum += colors * non_occluded
            non_occluded_sum += non_occluded

    # Step 3. 合并输出并写入文件
    if args.use_vertex_normal:
        v_colors = results['rgb_fine'].cpu().numpy() * 255.0
    else:  # 组合颜色是所有视图中的平均颜色
        v_colors = v_color_sum / non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr + v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertex_all, 'vertex'),
             PlyElement.describe(face, 'face')]).write(f"{args.scene_name}.ply")

    print('Done!')
