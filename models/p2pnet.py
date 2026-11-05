import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np
import time


def focal_alpha_factor(target_classes, alpha, foreground_class=1):
    """Return per-proposal alpha weights for background/foreground focal loss."""
    return torch.where(
        target_classes == foreground_class,
        torch.full_like(target_classes, alpha, dtype=torch.float32),
        torch.full_like(target_classes, 1.0 - alpha, dtype=torch.float32),
    )

# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)

# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]

# Point Relation Module (PRM): Neighbor-Aware Point Refinement via Self-Attention
# Innovation: allows each predicted point to attend to all other predictions,
# enabling collaborative refinement for structurally regular objects (e.g. corn kernels).
class PointRelationModule(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 num_classes=1, dropout=0.1, topk=0):
        super().__init__()
        self.d_model = d_model
        self.topk = int(topk or 0)
        if self.topk < 0:
            raise ValueError(f'PRM top-k must be non-negative, got {self.topk}')
        self.last_attention_tokens = None

        # Input projection: [offset_x, offset_y, max_score] -> d_model
        self.input_proj = nn.Linear(2 + 1, d_model)

        # Position encoding: anchor (x, y) -> d_model
        self.pos_enc = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads: predict residual corrections
        self.offset_head = nn.Linear(d_model, 2)
        self.class_head = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        # Small init for residual corrections to start near zero
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)

    @staticmethod
    def _normalize_positions(anchor_points):
        """Normalize anchors using the full image grid, before top-k selection."""
        pos_min = anchor_points.min(dim=1, keepdim=True).values
        pos_max = anchor_points.max(dim=1, keepdim=True).values
        return (anchor_points - pos_min) / (pos_max - pos_min + 1e-6)

    @staticmethod
    def _gather_tokens(tensor, indices):
        gather_index = indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
        return tensor.gather(1, gather_index)

    def _refine_tokens(self, offsets, scores, normalized_positions):
        """Run the unchanged PRM embedding/Transformer/heads on given tokens."""
        max_score = scores.max(dim=-1, keepdim=True).values
        point_feat = torch.cat([offsets, max_score], dim=-1)
        point_emb = self.input_proj(point_feat)
        pos_emb = self.pos_enc(normalized_positions)
        x = self.transformer(point_emb + pos_emb)
        return offsets + self.offset_head(x), scores + self.class_head(x)

    def forward(self, offsets, scores, anchor_points):
        """
        Args:
            offsets:     [B, N, 2] initial regression offsets
            scores:      [B, N, C] initial classification logits
            anchor_points: [B, N, 2] anchor point coordinates
        Returns:
            refined_offsets: [B, N, 2]
            refined_scores:  [B, N, C]
        """
        num_candidates = offsets.shape[1]
        normalized_positions = self._normalize_positions(anchor_points)

        # topk <= 0 preserves the original global-attention implementation.
        if self.topk <= 0 or self.topk >= num_candidates:
            self.last_attention_tokens = num_candidates
            return self._refine_tokens(offsets, scores, normalized_positions)

        if scores.shape[-1] < 2:
            raise ValueError('Top-k PRM requires background and foreground logits')

        # Rank by pre-PRM foreground probability.  This selection is used in
        # both training and inference; no ground-truth or test-time oracle is
        # involved.  Unselected candidates remain available to the final
        # detector/counting threshold and simply bypass PRM refinement.
        foreground_probability = torch.softmax(scores, dim=-1)[..., 1]
        selected_indices = torch.topk(
            foreground_probability, k=self.topk, dim=1, largest=True,
            sorted=False,
        ).indices

        selected_offsets = self._gather_tokens(offsets, selected_indices)
        selected_scores = self._gather_tokens(scores, selected_indices)
        selected_positions = self._gather_tokens(
            normalized_positions, selected_indices
        )
        refined_offsets, refined_scores = self._refine_tokens(
            selected_offsets, selected_scores, selected_positions
        )

        offset_index = selected_indices.unsqueeze(-1).expand(-1, -1, 2)
        score_index = selected_indices.unsqueeze(-1).expand(
            -1, -1, scores.shape[-1]
        )
        self.last_attention_tokens = self.topk
        return (
            offsets.scatter(1, offset_index, refined_offsets),
            scores.scatter(1, score_index, refined_scores),
        )


# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, use_prm=False,
                 prm_d_model=64, prm_nhead=4, prm_num_layers=2,
                 prm_dim_feedforward=128, prm_topk=0, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes + 1  # +1 for background class in classification
        self.use_prm = use_prm
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3,], row=row, line=line)

        self.fpn = Decoder(256, 512, 512)

        # Optional Point Relation Module
        if use_prm:
            self.prm = PointRelationModule(
                d_model=prm_d_model, nhead=prm_nhead,
                num_layers=prm_num_layers,
                dim_feedforward=prm_dim_feedforward,
                num_classes=self.num_classes,
                topk=prm_topk,
            )

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[1]) * 100 # 8x
        classification = self.classification(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)

        # Apply PRM refinement if enabled
        if self.use_prm:
            regression, classification = self.prm(regression, classification, anchor_points)

        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification

        out = {'pred_logits': output_class, 'pred_points': output_coord}

        return out

class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
                 use_smooth_l1=False, smooth_l1_beta=1.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            use_focal_loss: if True, use focal loss instead of standard cross-entropy
            use_smooth_l1: if True, use smooth L1 loss instead of MSE for point regression
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL or Focal Loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.use_focal_loss:
            # Focal Loss implementation
            num_classes = src_logits.shape[-1]
            # softmax over classes
            prob = src_logits.softmax(-1)  # [B, N, C]
            # gather the probability of the target class
            p_t = prob.gather(-1, target_classes.unsqueeze(-1)).squeeze(-1)
            # focal weight: alpha * (1 - p_t)^gamma
            alpha_t = focal_alpha_factor(target_classes, self.focal_alpha).to(src_logits.dtype)
            focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
            # cross entropy per element
            ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction='none')
            loss_ce = (focal_weight * ce_loss).sum() / num_points
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if self.use_smooth_l1:
            loss_bbox = F.smooth_l1_loss(src_points, target_points, reduction='none', beta=self.smooth_l1_beta)
        else:
            loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_points'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}
        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses

# create the P2PNet model
def build(args, training):
    # treats persons as a single class
    num_classes = 1

    backbone = build_backbone(args)
    use_prm = getattr(args, 'use_prm', False)
    model = P2PNet(backbone, args.row, args.line,
                   use_prm=use_prm,
                   prm_d_model=getattr(args, 'prm_d_model', 64),
                   prm_nhead=getattr(args, 'prm_nhead', 4),
                   prm_num_layers=getattr(args, 'prm_num_layers', 2),
                   prm_dim_feedforward=getattr(args, 'prm_dim_feedforward', 128),
                   prm_topk=getattr(args, 'prm_topk', 0),
                   num_classes=num_classes)
    if not training: 
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses,
                                use_focal_loss=getattr(args, 'use_focal_loss', False),
                                focal_alpha=getattr(args, 'focal_alpha', 0.25),
                                focal_gamma=getattr(args, 'focal_gamma', 2.0),
                                use_smooth_l1=getattr(args, 'use_smooth_l1', False),
                                smooth_l1_beta=getattr(args, 'smooth_l1_beta', 1.0))

    return model, criterion
