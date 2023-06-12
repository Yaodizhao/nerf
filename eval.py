import os
from argparse import ArgumentParser
from collections import defaultdict

import imageio
from tqdm import tqdm

from datasets import dataset_dict
from datasets.depth_utils import *
from models.nerf import *
from models.rendering import render_rays
from utils import load_ckpt, metrics

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='数据集的根目录')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='要验证的数据集')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='场景名称，用作输出文件夹名称')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[504, 378],
                        help='图像分辨率')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='图像是否以球形姿势拍摄（对于LLFF）')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='粗采样数量')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='额外精细采样数量')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='使用视差深度采样')
    parser.add_argument('--chunk', type=int, default=32 * 1024 * 16,
                        help='避免OOM而拆分的输入光线的块大小')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='要加载的预训练检查点路径')
    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='是否保存深度预测')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='要保存的格式')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """使用块对光线进行批量推理."""
    B = rays.shape[0]
    chunk = 1024 * 32 * 6
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i + chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()

        if args.save_depth:
            depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
            depth_pred = np.nan_to_num(depth_pred)
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(f'depth_{i:03d}', 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
