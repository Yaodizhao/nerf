import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        定义一个将 x 嵌入到 （x， sin（2^k x）， cos（2^k x）， ...）的函数
        in_channels: 输入通道数（xyz 和方向均为3个）
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            # 生成2的 0 次幂 到 2 的 N_freqs-1 次幂的张量
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        与论文不同，“x”也在输出中
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        D: 密度（sigma）编码器的层数
        W: 每层中的隐藏单元数
        in_channels_xyz：xyz 的输入通道数（默认为 3+3*10*2=63）
        in_channels_dir：方向输入通道数（默认为 3+3*4*2=27）
        skips：在第 D 层中添加跳过连接
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir, W // 2),
            nn.ReLU(True))

        # output layers 使用sigmoid激活函数将rgb限制在0，1之间
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
            nn.Linear(W // 2, 3),
            nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        将输入 （xyz+dir） 编码为 rgb+sigma（尚未准备好渲染）
        要渲染此光线，请参阅 rendering.py

        Inputs:
            x: (B, self.in_channels_xyz (+self.in_channels_dir))
               the embedded vector of position and direction
               经过位置编码的输入数据
            sigma_only: 是否仅推断sigma。 If True,
                        x is of shape (B, self.in_channels_xyz)


        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            # 在第五层重新添加xyz作为输入 (256 + 63)
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            # 利用反射机制进行数据进入nerf层
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        sigma = self.sigma(xyz_)  # (1024*32, 1)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)  # (1024*32, 128)
        rgb = self.rgb(dir_encoding)  # (1024*32, 3)

        out = torch.cat([rgb, sigma], -1)  # (1024*32, 4)

        return out
