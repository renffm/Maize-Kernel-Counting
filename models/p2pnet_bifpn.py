"""P2PNet baseline with its one-way FPN replaced by a single BiFPN layer."""

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import build_backbone
from .matcher import build_matcher_crowd
from .p2pnet import P2PNet
from .p2pnet_old import P2PNetOld, SetCriterion_Crowd


class FastNormalizedFusion(nn.Module):
    """Learn the relative contribution of every input feature."""

    def __init__(self, num_inputs, epsilon=1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, inputs):
        if len(inputs) != self.weights.numel():
            raise ValueError(
                f'Expected {self.weights.numel()} fusion inputs, got {len(inputs)}'
            )
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.epsilon)
        return sum(weight * feature for weight, feature in zip(weights, inputs))


class SeparableConvBlock(nn.Module):
    """Lightweight convolution applied after a BiFPN fusion node."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feature):
        feature = self.depthwise(feature)
        feature = self.pointwise(feature)
        return self.act(self.bn(feature))


class BiFPN(nn.Module):
    """Single-layer C3/C4/C5 bidirectional feature pyramid."""

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super().__init__()
        self.P3_in = nn.Conv2d(C3_size, feature_size, kernel_size=1)
        self.P4_in = nn.Conv2d(C4_size, feature_size, kernel_size=1)
        self.P5_in = nn.Conv2d(C5_size, feature_size, kernel_size=1)

        # Top-down nodes.
        self.P4_td_fusion = FastNormalizedFusion(2)
        self.P3_out_fusion = FastNormalizedFusion(2)
        self.P4_td_conv = SeparableConvBlock(feature_size)
        self.P3_out_conv = SeparableConvBlock(feature_size)

        # Bottom-up nodes. P4_out receives P3_out, so the P4 prediction feature
        # now contains high-resolution C3 information as well as C4 and C5.
        self.P4_out_fusion = FastNormalizedFusion(3)
        self.P5_out_fusion = FastNormalizedFusion(2)
        self.P4_out_conv = SeparableConvBlock(feature_size)
        self.P5_out_conv = SeparableConvBlock(feature_size)

    @staticmethod
    def _upsample(feature, target):
        return F.interpolate(feature, size=target.shape[-2:], mode='nearest')

    @staticmethod
    def _downsample(feature, target):
        feature = F.max_pool2d(feature, kernel_size=3, stride=2, padding=1)
        if feature.shape[-2:] != target.shape[-2:]:
            feature = F.interpolate(
                feature, size=target.shape[-2:], mode='nearest'
            )
        return feature

    def forward(self, inputs):
        if len(inputs) != 3:
            raise ValueError(f'BiFPN expects [C3, C4, C5], got {len(inputs)} inputs')
        C3, C4, C5 = inputs
        P3_in = self.P3_in(C3)
        P4_in = self.P4_in(C4)
        P5_in = self.P5_in(C5)

        # Top-down: semantic information flows from C5 to P3.
        P4_td = self.P4_td_conv(self.P4_td_fusion([
            P4_in,
            self._upsample(P5_in, P4_in),
        ]))
        P3_out = self.P3_out_conv(self.P3_out_fusion([
            P3_in,
            self._upsample(P4_td, P3_in),
        ]))

        # Bottom-up: high-resolution information flows back from P3 to P5.
        P4_out = self.P4_out_conv(self.P4_out_fusion([
            P4_in,
            P4_td,
            self._downsample(P3_out, P4_in),
        ]))
        P5_out = self.P5_out_conv(self.P5_out_fusion([
            P5_in,
            self._downsample(P4_out, P5_in),
        ]))
        return [P3_out, P4_out, P5_out]


class P2PNetBiFPN(P2PNetOld):
    """Original P2PNet heads/anchors/losses with only FPN replaced."""

    def __init__(self, backbone, row=2, line=2):
        super().__init__(backbone, row=row, line=line)
        self.fpn = BiFPN(256, 512, 512)


class P2PNetBiFPNPRM(P2PNet):
    """BiFPN P4 predictions refined by the existing point relation module."""

    def __init__(self, backbone, args):
        super().__init__(
            backbone, row=args.row, line=args.line, use_prm=True,
            prm_d_model=args.prm_d_model,
            prm_nhead=args.prm_nhead,
            prm_num_layers=args.prm_num_layers,
            prm_dim_feedforward=args.prm_dim_feedforward,
            prm_topk=getattr(args, 'prm_topk', 0),
            num_classes=1,
        )
        self.fpn = BiFPN(256, 512, 512)


def build(args, training):
    if getattr(args, 'use_focal_loss', False):
        raise ValueError('P2PNet-BiFPN variants do not use focal loss')
    num_classes = 1
    backbone = build_backbone(args)
    if getattr(args, 'use_prm', False):
        model = P2PNetBiFPNPRM(backbone, args)
    else:
        model = P2PNetBiFPN(backbone, args.row, args.line)
    if not training:
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        use_focal_loss=False,
        use_smooth_l1=False,
    )
    return model, criterion
