"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pdb import set_trace as stx

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class MultiHeadCASAtt(nn.Module):
    def __init__(self, dim, num_heads=1, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(self.head_dim),
            ChannelOperation(self.head_dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(self.head_dim),
            ChannelOperation(self.head_dim),
        )
        self.dwc = nn.Conv2d(self.head_dim, self.head_dim, 3, 1, 1, groups=self.head_dim)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3), qkv)

        q = q.reshape(B * self.num_heads, self.head_dim, H * W).transpose(1,2)
        k = k.reshape(B * self.num_heads, self.head_dim, H * W).transpose(1,2)
        v = v.reshape(B * self.num_heads, self.head_dim, H * W).transpose(1,2)

        # q = self.oper_q(q).reshape(B, self.num_heads, self.head_dim, H * W)
        # k = self.oper_k(k).reshape(B, self.num_heads, self.head_dim, H * W)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # out = (attn @ v.transpose(-2, -1)).transpose(2, 3).reshape(B, C, H, W)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


class GraphOperation(nn.Module):
    def __init__(self, dim, k=9):  # k 表示每个节点的邻居数
        super().__init__()
        self.k = k
        self.gcn = GCNConv(dim, dim)  # 图卷积层

    def build_graph(self, x):
        """
        构建图结构，返回邻接矩阵。
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        Returns:
            edge_index (torch.Tensor): 邻接矩阵，形状为 (2, E)，E 是边的数量。
        """
        B, C, H, W = x.shape
        N = H * W  # 总节点数
        # N = W
        edge_index = []

        # 创建每个节点与其 k 个最近邻节点的连接关系
        for i in range(N):
            neighbors = list(range(max(0, i - self.k), min(N, i + self.k + 1)))
            neighbors.remove(i)  # 去掉自身
            for j in neighbors:
                edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = edge_index.to(x.device)
        return edge_index

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 将特征图展平为 (B, C, N)，然后转置为 (B, N, C)
        x = x.view(B, C, -1).transpose(1, 2).reshape(B, C, H, W)

        # 构建图结构
        edge_index = self.build_graph(x)

        # 图卷积操作
        x = x.view(B, C, -1).transpose(1, 2)
        x = self.gcn(x, edge_index)

        # 转回原始形状
        x = x.transpose(1, 2).view(B, C, H, W)

        return x

###########################################################################
# #todo ContraNorm 类
# class ContraNorm(nn.Module):
#     def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
#         super().__init__()
#         if learnable and scale > 0:
#             import math
#             if positive:
#                 scale_init = math.log(scale)
#             else:
#                 scale_init = scale
#             self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
#         self.dual_norm = dual_norm
#         self.scale = scale
#         self.pre_norm = pre_norm
#         self.temp = temp
#         self.learnable = learnable
#         self.positive = positive
#         self.identity = identity
#
#         self.layernorm = nn.LayerNorm(dim, eps=1e-6)
#
#     def forward(self, x):
#         if self.scale > 0.0:
#             xn = nn.functional.normalize(x, dim=2)
#             if self.pre_norm:
#                 x = xn
#             sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
#             if self.dual_norm:
#                 sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
#             else:
#                 sim = nn.functional.softmax(sim, dim=2)
#             x_neg = torch.bmm(sim, x)
#             if not self.learnable:
#                 if self.identity:
#                     x = (1 + self.scale) * x - self.scale * x_neg
#                 else:
#                     x = x - self.scale * x_neg
#             else:
#                 scale = torch.exp(self.scale_param) if self.positive else self.scale_param
#                 scale = scale.view(1, 1, -1)
#                 if self.identity:
#                     x = scale * x - scale * x_neg
#                 else:
#                     x = x - scale * x_neg
#         x = self.layernorm(x)
#         return x

##############################################################################

class GNNTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attn_bias=False, drop=0., proj_drop=0., k=9):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        # self.norm1 = ContraNorm(dim=dim)

        self.gnn = GraphOperation(dim, k=k)  # 使用 GNN 进行特征聚合

        self.norm2 = nn.LayerNorm(dim)
        # self.norm2 = ContraNorm(dim=dim)

        self.attn = MultiHeadCASAtt(dim, num_heads, attn_bias, proj_drop) #todo
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, drop)

    def forward(self, x):
        B, C, H, W = x.shape

        # 图神经网络特征聚合
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        test = self.norm1(x).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.gnn(test).flatten(2).transpose(1, 2)

        x = x + self.attn(self.norm1(x).transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2) #todo
        test2 = self.norm2(x).transpose(1, 2).reshape(B, C, H, W)
        # 前馈网络增强特征
        x = x + self.mlp(test2).flatten(2).transpose(1, 2)

        # 转回原始形状
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attn_bias=False, drop=0., proj_drop=0., depth=2, k=9):
        super().__init__()
        # self.adjust_dim = nn.Conv2d(3,dim,1,1,0) #todo
        self.norm1 = nn.LayerNorm(dim)
        # self.norm1 = ContraNorm(dim=dim)

        self.attn = MultiHeadCASAtt(dim, num_heads, attn_bias, proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        # self.norm2 = ContraNorm(dim=dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, drop)

        #todo 新加如下代码
        self.embed_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)  # 输入通道扩展
        self.transformer_blocks = nn.ModuleList([
            GNNTransformerBlock(dim, num_heads, mlp_ratio, k=k) for _ in range(depth)
        ])
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)  # 输出通道还原

    def forward(self, x):
        B, C, H, W = x.shape

        #todo 新代码如下修改
        # 输入通道扩展
        x = self.embed_conv(x)
        # Transformer 层
        for block in self.transformer_blocks:
            x = block(x)
        # 输出通道还原
        x = self.out_conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = x + self.attn(self.norm1(x).transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2)
        x = x + self.mlp(self.norm2(x).transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

#############################################
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1,
         dilation=1):  # todo 创建一个带有自动填充的二维卷积层,kernel_size为卷积核大小,dilation为膨胀率
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=autopad(kernel_size, None, dilation), bias=bias, stride=stride, dilation=dilation)
    # padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
##########################################################################
## Multi-Scale Channel Attention Module
class MultiScaleChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, scales=[1, 2, 4], bias=False):
        super(MultiScaleChannelAttention, self).__init__()
        self.scales = scales  # ，， AdaptiveAvgPool2d(scale)
        self.avg_pools = nn.ModuleList(
            [nn.AvgPool2d(kernel_size=(scale, scale), stride=(scale, scale)) for scale in scales])
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel * len(scales), channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        pooled_features = []
        original_height, original_width = x.shape[2], x.shape[3]
        for avg_pool in self.avg_pools:
            pooled_feature = avg_pool(x)
            # Upsample the pooled feature to match the original input size
            upsampled_feature = F.interpolate(pooled_feature, size=(original_height, original_width), mode='bilinear',
                                              align_corners=False)
            pooled_features.append(upsampled_feature)

        # Concatenate pooled features from different scales
        y = torch.cat(pooled_features, dim=1)
        # Generate channel attention weights
        y = self.conv_du(y)
        return x * y

## Multi-Scale Spatial Attention Module
class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, channel, kernel_sizes=[3, 5, 7], bias=False):
        super(MultiScaleSpatialAttention, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
                nn.Sigmoid()
            ) for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        spatial_attentions = [conv(x) for conv in self.convs]
        y = torch.sum(torch.stack(spatial_attentions, dim=-1), dim=-1)
        return x * y


## Combined Channel and Spatial Attention Layer
class CALayerforEncoder(nn.Module):
    def __init__(self, channel, reduction=16, scales=[1, 2, 4], kernel_sizes=[3, 5, 7], bias=False):
        super(CALayerforEncoder, self).__init__()
        self.channel_attention = MultiScaleChannelAttention(channel, reduction=reduction, scales=scales, bias=bias)
        self.spatial_attention = MultiScaleSpatialAttention(channel, kernel_sizes=kernel_sizes, bias=bias)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# class TiedSELayer(nn.Module):
#     '''Tied Block Squeeze and Excitation Layer'''
#
#     def __init__(self, channel, B=1, reduction=16):
#         super(TiedSELayer, self).__init__()
#         assert channel % B == 0
#         self.B = B
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         channel = channel // B
#         self.fc = nn.Sequential(
#             nn.Linear(channel, max(2, channel // reduction)),
#             nn.ReLU(inplace=True),
#             nn.Linear(max(2, channel // reduction), channel),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b * self.B, c // self.B)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
#
# class TiedBlockConv2d(nn.Module):
#     '''Tied Block Conv2d'''
#
#     def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=True, \
#                  B=1, args=None, dropout_tbc=0.0, groups=1):
#         super(TiedBlockConv2d, self).__init__()
#         assert planes % B == 0
#         assert in_planes % B == 0
#         self.B = B
#         self.stride = stride
#         self.padding = padding
#         self.out_planes = planes
#         self.kernel_size = kernel_size
#         self.dropout_tbc = dropout_tbc
#         self.conv = nn.Conv2d(in_planes // self.B, planes // self.B, kernel_size=kernel_size, stride=stride, \
#                               padding=padding, bias=bias, groups=groups)
#         if self.dropout_tbc > 0.0:
#             self.drop_out = nn.Dropout(self.dropout_tbc)
#
#     def forward(self, x):
#         n, c, h, w = x.size()
#         x = x.contiguous().view(n * self.B, c // self.B, h, w)
#         h_o = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
#         w_o = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
#         x = self.conv(x)
#         x = x.view(n, self.out_planes, h_o, w_o)
#         if self.dropout_tbc > 0:
#             x = self.drop_out(x)
#         return x

class CABforEncoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CABforEncoder, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayerforEncoder(n_feat, reduction, bias=bias)  # 使用 CALayer 实现多尺度通道注意力机制

        self.body = nn.Sequential(*modules_body)  # 创建一个包含两个卷积层和激活函数的标准卷积模块 body
        # self.elgca = ELGCA(dim=n_feat, heads=4)
        # self.tiedblockconv2d = TiedBlockConv2d(in_planes=n_feat, planes=n_feat, kernel_size=3, stride=1, padding=1)
        # self.TBC = TiedSELayer(n_feat)

    def forward(self, x):  # 先通过 body 处理输入特征图，然后应用通道注意力机制，最后加上残差连接
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        res = self.body(x)
        ################################################
        # res = self.tiedblockconv2d(res)
        # res = self.TBC(res)
        ################################################

        res = self.CA(res)
        res += x
        return res


###############################################上述为CAB的修改代码

## Channel Attention Layer，，通道注意力机制
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point，使用全局平均池化将特征图压缩为单个点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight，生成通道注意力权重
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y  # 生成的注意力权重乘以原始特征图，得到加权后的特征图


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)  # 使用 CALayer 实现通道注意力机制

        self.body = nn.Sequential(*modules_body)  # 创建一个包含两个卷积层和激活函数的标准卷积模块 body

    def forward(self, x):  # 先通过 body 处理输入特征图，然后应用通道注意力机制，最后加上残差连接
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, channel):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, channel, kernel_size, bias=bias)
        self.conv3 = conv(channel, n_feat, kernel_size, bias=bias)
        self.transformer = TransformerBlock(dim=3, num_heads=1)

    def forward(self, x, x_img):  # x为深层特征,x_img为浅层特征
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img

        img =  F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
        img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
        img = self.transformer(img)
        img = F.interpolate(img, scale_factor=2.0, mode='bilinear', align_corners=False)
        img = F.interpolate(img, scale_factor=2.0, mode='bilinear', align_corners=False)

        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()
        # 创建三个不同级别的特征提取模块列表，分别处理不同维度的特征图，1：主要负责处理相对原始的特征信息，保持较高的空间分辨率，2：特征图的通道数增加了，意味着可以学习更复杂的特征表示，3：进一步增加
        self.encoder_level1 = [CABforEncoder(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CABforEncoder(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _
                               in range(2)]
        self.encoder_level3 = [CABforEncoder(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()  # 将输入张量的高度和宽度缩小一半，并使用双线性插值方法进行缩放
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
##########################################todo 如下为新加入安全通道注意力
def gram_schmidt(filters):
    """
    Apply Gram-Schmidt process to make filters orthogonal.
    """
    n, c, h, w = filters.shape
    filters = filters.view(n, -1)  # Reshape to [n, c*h*w]
    q = []
    for i in range(n):
        v = filters[i]
        for j in range(i):
            v = v - torch.dot(v, q[j]) * q[j]
        v = v / torch.norm(v)
        q.append(v)
    q = torch.stack(q, dim=0).view(n, c, h, w)
    return q

class OrthogonalChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(OrthogonalChannelAttention, self).__init__()
        # Initialize random filters and apply Gram-Schmidt process
        filters = torch.randn(in_channels, 1, 1, 1)  # Shape [in_channels, 1, 1, 1]
        self.filters = nn.Parameter(gram_schmidt(filters), requires_grad=False)
        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze: Apply orthogonal filters
        attention = self.excitation(x)
        # Scale
        return x * attention
##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.oca = OrthogonalChannelAttention(n_feat + scale_orsnetfeats)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])
        x = self.oca(x) #todo

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))
        x = self.oca(x) #todo

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        x = self.oca(x) #todo 新加入

        return x

import matplotlib.pyplot as plt
import numpy as np
def save_heatmap(features, filename):
    """
    生成并保存特征图的热力图
    features: 输入特征图（B, C, H, W）
    filename: 保存的文件名
    """
    # 假设只对第一个图像进行可视化
    feature_map = features[0].cpu().detach().numpy()  # 取出第一个图像并转为numpy

    # 对特征图进行归一化处理
    feature_map = np.mean(feature_map, axis=0)  # 对所有通道取平均，得到单通道图像
    feature_map = np.maximum(feature_map, 0)  # 设置负值为0
    feature_map = feature_map / feature_map.max()  # 归一化到[0, 1]区间

    # 显示热力图
    plt.imshow(feature_map, cmap='jet')
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

##########################################################################
class MPRNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(MPRNet, self).__init__()
        # self.memory = MemModule(ptt_num=2, num_cls=10, part_num=5, fea_dim=in_c)  # todo
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                    num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, channel=in_c)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, channel=in_c)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img, isTrain=False):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)
        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        res1_img = torch.cat([res1_top[0], res1_bot[0]], 2)
        save_heatmap(res1_img, 'res1_img_feature_map.png')

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        save_heatmap(stage1_img, 'stage1_img_feature_map.png')

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)  # todo
        save_heatmap(res2[0], 'res2[0]_feature_map.png')
        save_heatmap(stage2_img, 'stage2_img_feature_map.png')
        # x3_samfeats, stage2_img = self.sam23(res2[0], x1)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)
        save_heatmap(stage3_img, 'stage3_img_feature_map.png')
        n = [stage3_img + x3_img, stage2_img, stage1_img]  # todo 原论文 x3_img为 x1
        # n = [stage3_img + x1, stage2_img, stage1_img]
        if (isTrain):
            return n
        else:
            return 1, torch.clamp(n[0], 0, 1)