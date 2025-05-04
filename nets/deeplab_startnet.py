import torch
import torch.nn as nn
from timm.models.layers import DropPath


class StarNet(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=False, version='s050'):
        super().__init__()
        # 通道压缩系数
        self.width_ratio = 0.5  # s050的核心修改

        # 根据下采样率调整stage结构
        self.stage_config = self._adjust_stages(downsample_factor)

        # 初始化基础网络结构（通道数按比例缩小）
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(32 * self.width_ratio), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * self.width_ratio)),
            nn.ReLU6(inplace=True)
        )

        # 构建轻量化特征提取阶段
        self.stage1 = self._make_stage(int(32 * self.width_ratio), int(24 * self.width_ratio),
                                       self.stage_config[0], stride=2)
        self.stage2 = self._make_stage(int(24 * self.width_ratio), int(128 * self.width_ratio),
                                       self.stage_config[1], stride=2)
        self.stage3 = self._make_stage(int(128 * self.width_ratio), int(256 * self.width_ratio),
                                       self.stage_config[2], stride=2)
        self.stage4 = self._make_stage(int(256 * self.width_ratio), int(320 * self.width_ratio),
                                       self.stage_config[3], stride=1)
        self.adjust_channles_96 = nn.Conv2d(160, 96, 1)

        # 加载预训练权重（需要适配s050结构）
        if False:
            self._load_pretrained(version)

        # 通道数配置（适配MinneApple的两类分割）
        self.low_level_channels = int(24 * self.width_ratio)
        self.aspp_channels = int(320 * self.width_ratio)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        """ 轻量化stage构建 """
        layers = []
        # 下采样层
        if stride == 2:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU6(inplace=True))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))

        # 使用更少的StarBlock
        for _ in range(num_blocks):
            layers.append(StarBlock(out_ch))
        return nn.Sequential(*layers)

    def _adjust_stages(self, downsample_factor):
        """ s050专用配置（减少block数量） """
        if downsample_factor == 8:
            return [1, 1, 4, 2]  # 原s1配置的1/2
        elif downsample_factor == 16:
            return [2, 2, 6, 3]  # 原s4配置的1/2
        else:
            raise ValueError(f"Unsupported downsample_factor: {downsample_factor}")

    def _load_pretrained(self, version):
        """ 加载适配s050的预训练权重 """
        if version != 's050':
            raise ValueError("s050 version requires custom pretrained weights")

        # 这里需要实际预训练路径，暂时用随机初始化
        print("s050使用随机初始化（需提供专用预训练权重）")

    def forward(self, x):
        # 前向传播保持结构不变
        x = self.stem(x)
        low_level_feat = self.stage1(x)
        x = self.stage2(low_level_feat)
        x = self.stage3(x)
        x_aspp_before = self.stage4(x)
        x_aspp_before = self.adjust_channles_96(x_aspp_before)
        return low_level_feat, x_aspp_before


class StarBlock(nn.Module):
    """ 轻量化StarBlock（保持结构，减少内部通道） """

    def __init__(self, dim):
        super().__init__()
        # 保持结构，通道数由外部控制
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=False),  # 扩展比例从4x降为2x
            nn.BatchNorm2d(dim * 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return x + self.act(x) * (self.act(self.proj(self.dwconv(x))))

    #   super().__init__()
    #         # 保持结构，通道数由外部控制
    #         self.dwconv = nn.Sequential(
    #             nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
    #             nn.BatchNorm2d(dim),
    #             nn.ReLU6(inplace=True),
    #
    #         )
    #         self.f1 = nn.Sequential(
    #             nn.Conv2d(dim, dim * 2, 1, bias=False),  # 扩展比例从4x降为2x
    #             nn.BatchNorm2d(dim * 2),
    #             nn.ReLU6(inplace=True),
    #             nn.Conv2d(dim * 2, dim, 1, bias=False),
    #             nn.BatchNorm2d(dim)
    #         )
    #
    #
    #
    #         self.mish = nn.Mish()  # 正确实例化SELU模块
    #
    #     def forward(self, x):
    #         x = self.dwconv(x)
    #
    #         x1, x2 = self.f1(x), self.f1(x)
    #         x3 = self.mish(x1) * x2
    #
    #         return x + x3


# 测试输出
if __name__ == "__main__":
    model = StarNet(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    low, deep = model(x)
    print(f"浅层特征尺寸: {low.shape}")  # [2, 12, 128, 128]
    print(f"深层特征尺寸: {deep.shape}")  # [2, 160, 64, 64]
