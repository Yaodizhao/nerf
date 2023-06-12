import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/dev08/mega-nerf-data/yby',
                        help='数据集的根目录')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='训练的数据集类型')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[2048, 1365],
                        help='图像的分辨率（img_w、img_h）')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='图像是否以球形姿势拍摄（对于LLFF）')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='粗采样数量')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='额外精细采样数量')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='使用视差深度采样')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='扰动深度采样因子')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='正则化sigma所添加的噪声标准差')

    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='使用的损失函数')

    parser.add_argument('--batch_size', type=int, default=2048,
                        help='批量大小')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='避免OOM而拆分的输入光线的块大小')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='训练波数')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='gpu数')
    parser.add_argument('--list_gpus', type=str, default="0,1,2,3",
                        help='gpu设备')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='要加载的预训练检查点路径')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='检查点状态字典中要忽略的前缀')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='优化器类型',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='加速迭代收敛的学习率动量')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='权重衰减')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='学习率调度器(StepLR, CosineAnnealingLR, PolynomialLR)',
                        choices=['steplr', 'cosine', 'poly'])

    # 用于预热的参数，仅在优化器 == 'SGD' 或 'adam' 时应用
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='当multiplier=1.0时，学习率lr从0开始增到base_lr为止'
                             '当multiplier大于1.0时，学习率lr从base_lr开始增到base_lr*multiplier为止'
                             'multiplier不能小于1.0。')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='优化器中逐渐预热（增加）学习率')

    # steplr学习率调度器的参数
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='学习率衰减周期')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='学习率衰减系数')

    # poly学习率调度器的参数
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='多项式学习率衰减的幂指数')

    parser.add_argument('--exp_name', type=str, default='yby',
                        help='实验名称')

    return parser.parse_args()
