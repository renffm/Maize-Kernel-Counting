
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher_Crowd(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].softmax(-1)  # [bs, num_queries, num_classes]
        out_points = outputs["pred_points"]  # [bs, num_queries, 2]

        indices = []
        # build per-batch cost and solve assignment per sample
        for b in range(bs):
            tgt_pts = targets[b]["point"]
            tgt_ids = targets[b]["labels"]

            # classification cost: [num_queries, num_targets]
            cost_class = -out_prob[b][:, tgt_ids]

            # point distance cost
            if tgt_pts.numel() == 0:
                # no targets in this image
                C = cost_class.new_zeros((num_queries, 0))
            else:
                cost_point = torch.cdist(out_points[b], tgt_pts, p=2)
                C = self.cost_point * cost_point + self.cost_class * cost_class

            C_np = C.cpu().numpy()
            if C_np.size == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
            else:
                i, j = linear_sum_assignment(C_np)
                indices.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))

        return indices


def build_matcher_crowd(args):
    return HungarianMatcher_Crowd(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
