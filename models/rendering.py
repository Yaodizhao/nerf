import torch
from torchsearchsorted import searchsorted

__all__ = ['render_rays']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    样本 @N_importance 来自分布由 @weights 定义的 @bins 样本

    Inputs:
        bins: (N_rays, N_samples_+1) 其中 N_samples_ 是每条射线的粗样品数 - 2
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not 扰动深度采样因子是否为0
        eps: 较小的数字以防止被零除

    Outputs:
        samples: the sampled samples
    """

    # 根据 pdf 求 cdf
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # 防止除以零 (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples) 不透明度的累积分布函数
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # 填充至 0~1（含 0~1）

    # 随机生成[0,1]区间的采样点
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    # 获取采样点在 CDF 中的索引值 使用below和above包围起来
    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds - 1, 0)
    # below = torch.clamp_min(inds - torch.ones_like(torch.empty(inds.shape[0], inds.shape[1])), 0)
    above = torch.clamp_max(inds, N_samples_)
    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)  # ( n1*n2*c -> n1*(n2 c) ) c=2

    # 将采样点在CDF中的前后两个索引值所对应的采样点密度区间(cdf_g),和对应的[0,1]选择的采样点的位置区间(bin_g)提取出来
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)  # ( n1*(n2 c) -> n1*n2*c ) c=2
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)  # ( n1*(n2 c) -> n1*n2*c ) c=2

    # 选择采样点密度分布函数区间之间密度变化很高的区间(即权重大的区间)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # DENOM 等于 0 表示 bin 的权重为 0，在这种情况下，不会对其进行采样
    # 因此设置为任何值都可以（此处设置为 1）
    denom[denom < eps] = 1

    # t =  (u - cdf_g[..., 0])/( cdf_g[..., 1] - cdf_g[..., 0])
    # 起始采样点 + (结束采样点-起始采样点) * t 线性插值
    # 为啥不用 torch.lerp( bins_g[..., 0], bins_g[..., 1], t)
    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False
                ):
    """
    通过使用rays计算model的输出来渲染光线

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
        models：neRF.py中定义的NeRF模型（粗略和精细）列表
        embeddings：nerf.py中定义的原点和方向的嵌入模型列表
        rays：（N_rays，3+3+2），射线原点、方向以及远近深度边界
        N_samples：每条射线的粗略采样数
        use_disp：是否在视差空间中采样（反向深度）
        perturb：扰动光线上采样位置的因子（仅适用于粗略模型）
        noise_std：扰动西格玛模型预测的因素
        N_importance：每条射线的精细采样数
        chunk：批处理推理中的chunk大小
        white_back：背景是否为白色（取决于数据集）
        test_time：是否为测试（仅限推理）。如果为True，则不会对粗略rgb进行推理以节省时间

    Outputs:
        结果：包含粗模型和精细模型的最终RGB和深度图的字典
    """

    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        """
        Helper function that performs model inference.
        执行模型推理的帮助函数。

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

            model：NeRF型号（粗略或精细）
            embeddingxyz：xyz的嵌入模块
            xyz_：（N_rays, N_samples_, 3）采样位置
                N_samples_是每条射线中采样点的数量
                    = N_samples (粗略模型)
                    = N_samples+N_importance (精细模型)
            dir_：（N_rays，3）射线方向
            dir_embedded:（N_rays，embed_dir_channels）嵌入方向
            z_vals：（N_rays，N_samples_）采样位置的深度
            weights_only:是否只对sigma进行推理

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): 每个采样的权重
            else:
                rgb_final: (N_rays, 3) 最终的rgb图像
                depth_final: (N_rays) 深度图
                weights: (N_rays, N_samples_): 每个采样的权重
        """
        N_samples_ = xyz_.shape[1]
        # 嵌入方向
        xyz_ = xyz_.view(-1, 3)  # (N_rays, N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
            # 复制方向位置编码 使得 每条光线对应64个采样点的方向位置编码相同 (N_rays*N_samples_, embed_dir_channels)

        # 执行模型推理以获取RGB和原始sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # 按块进行位置编码
            xyz_embedded = embedding_xyz(xyz_[i:i + chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i + chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        # N_rays / chunk 块数据拼接起来
        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)  # 取出模型预测的值 再变回(N_rays, N_samples_, 4)张量
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)

        # 使用体积渲染转换这些值(Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1) 每个采样点之间的距离
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) 最后一个delta是无穷大
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # 将每个距离乘以其相应方向射线的范数(长度)以转换为真实世界的距离（考虑非单位方向）。
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        # 添加噪声
        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # 通过公式计算alpha (3) α = 1 - e^(-δi*σi)
        # 添加relu 是因为密度不可能为负值 因此如果出现负值变成0
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, 1-a1, 1-a2, ...]

        # 不透明度计算cumprod是生成
        # [1,    (1-a1),   (1-a1)*(1-a2),   (1-a1)*(1-a2)*(1-a3), ...,   (1-a1)*(...)*(1-an-1)]
        # 再乘以 alphas 变成 weights 即
        # [a1, a2(1-a1), a3(1-a1)*(1-a2), a4(1-a1)*(1-a2)*(1-a3), ..., an(1-a1)*(...)*(1-an-1)]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]  # (N_rays, N_samples_)

        weights_sum = weights.sum(1)  # (N_rays), 将沿光线累积的不透明度累加起来
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # 计算最终加权输出
        # (N_rays, 3) unsqueeze是为了多一个轴便于相乘 将每个采样点透明度和每个点rgb相乘累加得到最终rgb值
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
        # 将每个采样点透明度和每个点深度相乘累加得到最终深度值 与rgb相同 只是把rgb换成z
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights

    # 从列表中提取粗模型以及xyz和方向的位置编码维度
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # 分解输入
    N_rays = rays.shape[0]  # 每块的光线个数 32 * 1024
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # (N_rays, 3) (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # (N_rays, 1) (N_rays, 1)

    # 嵌入方向
    dir_embedded = embedding_dir(rays_d)  # (N_rays, 27) 27 = 3(ray_d) + 3 (ray_d) * 4 (dir的维度是4) * 2 (sin cos)

    # 采样深度点
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # 粗网络首次均匀采样 共N_samples个样本
    if not use_disp:  # 在深度空间中使用线性采样
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # 在视差空间中使用线性抽样
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # 前63个和后63个取平均值(N_rays, 63) 选择间隔中点
        # 获取样本之间的间隔 补上头尾变成64个采样点
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        # 添加采样深度扰动 (z_vals)
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # rays_o (N_rays, 3) -> (N_rays, 1, 3)
    # rays_d (N_rays, 3) -> (N_rays, 1, 3)
    # z_vals (N_rays, N_samples) -> (N_rays, N_samples, 1)
    # rays_o*rays_d*z_vals  -> (N_rays, N_samples, 3)

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3) -> (32*1024, 64, 3)

    if test_time:
        weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        # 进行预测
        rgb_coarse, depth_coarse, weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                  }

    if N_importance > 0:  # 精细模型的采样点
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) 间隔中点
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # 使用detach，以便梯度不会从这里传播到weights_coarse
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
        # (N_rays, N_samples+N_importance, 3)
        # 精细网络训练
        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result
