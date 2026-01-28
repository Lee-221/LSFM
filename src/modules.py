import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 图像特征增强模块 (FMB) ====================
class ChannelMLP(nn.Module):
    """两层MLP进行通道间信息交互"""

    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Conv2d(dim, dim, 1, 1, 0)  # L1
        self.linear2 = nn.Conv2d(dim, dim, 1, 1, 0)  # L2

    def forward(self, x):
        # 公式(10)
        return self.linear2(torch.sigmoid(self.linear1(x)))


class ImageFeatureEnhancement(nn.Module):
    """图像特征增强模块 (3.2)"""

    def __init__(self, dim, down_scale=8, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.down_scale = down_scale
        self.eps = eps

        # 双分支结构
        self.linear_split = nn.Conv2d(dim, dim * 2, 1, 1, 0)

        # 空间分支组件
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.linear_w = nn.Conv2d(dim, dim, 1, 1, 0)
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        # 通道分支组件
        self.channel_mlp = ChannelMLP(dim)

        # 融合线性层
        self.linear_fuse = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, f):
        # L2范数归一化 (公式1)
        norm = torch.norm(f, p=2, dim=1, keepdim=True)
        f_norm = f / (norm + self.eps)

        # 双分支结构处理
        xy = self.linear_split(f_norm)
        x, y = xy.chunk(2, dim=1)  # F_x, F_y
        b, c, h, w = x.shape

        # 确保 h, w 是整数
        h_int, w_int = int(h), int(w)

        # 空间分支处理 (公式3-5)
        mu = torch.mean(x, dim=(-2, -1), keepdim=True)  # μ
        f_v = torch.mean((x - mu) ** 2, dim=(-2, -1), keepdim=True)  # F_v

        x_down = F.adaptive_max_pool2d(x, (h_int // self.down_scale, w_int // self.down_scale))
        f_s = self.dw_conv(x_down)

        w_tensor = torch.sigmoid(self.linear_w(self.alpha * f_s + self.beta * f_v))  # W
        w_up = F.interpolate(w_tensor, size=(h_int, w_int), mode='nearest')  # ✅ 使用整数
        f_l = x * w_up  # 局部增强特征 (公式5)

        # 通道分支处理 (公式10)
        f_c = self.channel_mlp(y)

        # 残差连接融合 (公式11-12)
        fused = self.linear_fuse(f_l + f_c)
        return fused + f  # 残差连接


# ==================== 多尺度轻量化特征提取模块 ====================
class MultiScaleFeatureExtraction(nn.Module):
    """多尺度轻量化特征提取模块 (3.3)"""

    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim
        chunk_dim = dim // n_levels

        # 多尺度特征提取 (公式13)
        self.mfr = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim),  # 深度卷积
                nn.Conv2d(chunk_dim, chunk_dim, 1),  # 逐点卷积
                nn.BatchNorm2d(chunk_dim)
            ) for _ in range(self.n_levels)])

        # 特征融合
        self.aggr = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )
        self.act = nn.GELU()

        # 动态特征融合组件 (公式15-19)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, groups=dim * 2),
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim)
        )

        # 空间注意力
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, groups=dim, bias=True),
            nn.Conv2d(dim, 1, 1, 1, bias=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, groups=dim, bias=True),
            nn.Conv2d(dim, 1, 1, 1, bias=True)
        )
        self.nonlin = nn.Sigmoid()


    def forward_safm(self, x):
        """多尺度特征提取 (公式13-14)"""
        h, w = x.size()[-2:]
        h_int, w_int = int(h), int(w)  #转换为整数
        xc = x.chunk(self.n_levels, dim=1)
        out = []

        for i in range(self.n_levels):
            if i > 0:  # 第i>0个尺度组构建金字塔结构
                p_size = (h_int // 2 ** i, w_int // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)  # 自适应最大池化
                s = self.mfr[i](s)  # 深度可分离卷积
                s = F.interpolate(s, size=(h_int, w_int), mode='bilinear')  # ✅ 使用整数
            else:  # 基础特征组直接提取
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        return self.act(out) * x  # 保留原始特征细节

    def forward_dff(self, x, skip):
        """动态特征融合 (公式15-19)"""
        output = torch.cat([x, skip], dim=1)

        # 通道注意力
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        # 空间注意力
        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        return output * att

    def forward(self, x, skip=None):
        x = self.forward_safm(x)
        if skip is not None:
            skip = self.forward_safm(skip)
            x = self.forward_dff(x, skip)
        return x


# ==================== 自注意力权重分配模块 ====================
class SelfAttentionWeightAllocation(nn.Module):
    """自注意力权重分配模块 (对应文档3.4节)"""

    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, 1, 1)

    def forward(self, x):
        # 通道分组处理 (公式20)
        b, c, h, w = x.size()
        h_int, w_int = int(h), int(w)  # 🔥 转换为整数

        group_x = x.reshape(b * self.groups, -1, h_int, w_int)  # ✅ 使用整数

        # 空间方向分解策略 (公式21-23)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h_int, w_int], dim=2)  # ✅ 使用整数

        # 双向特征交互
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # 注意力权重计算
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # 残差连接输出 (公式29)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w) + x