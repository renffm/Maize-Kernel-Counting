import unittest
from argparse import Namespace

import torch
from torch import nn

from models.p2pnet import PointRelationModule
from models.p2pnet_bifpn import BiFPN, P2PNetBiFPN, P2PNetBiFPNPRM


class DummyBackbone(nn.Module):
    def forward(self, image):
        batch_size, _, height, width = image.shape
        device = image.device
        return [
            torch.randn(batch_size, 128, height // 2, width // 2, device=device),
            torch.randn(batch_size, 256, height // 4, width // 4, device=device),
            torch.randn(batch_size, 512, height // 8, width // 8, device=device),
            torch.randn(batch_size, 512, height // 16, width // 16, device=device),
        ]


class BiFPNTest(unittest.TestCase):
    def test_prm_variant_keeps_p4_proposals_and_backpropagates(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = Namespace(
            row=2, line=2, prm_d_model=64, prm_nhead=4,
            prm_num_layers=2, prm_dim_feedforward=128, prm_topk=32,
        )
        baseline = P2PNetBiFPN(DummyBackbone()).to(device).eval()
        model = P2PNetBiFPNPRM(DummyBackbone(), args).to(device).train()
        image = torch.randn(1, 3, 64, 64, device=device)

        with torch.no_grad():
            baseline_output = baseline(image)
        output = model(image)
        for key in ('pred_logits', 'pred_points'):
            self.assertEqual(output[key].shape, (1, 256, 2))
            self.assertEqual(output[key].shape, baseline_output[key].shape)

        (output['pred_logits'].square().mean()
         + output['pred_points'].square().mean()).backward()
        self.assertIsNotNone(model.prm.offset_head.weight.grad)
        self.assertIsNotNone(model.fpn.P3_in.weight.grad)
        self.assertEqual(model.prm.last_attention_tokens, 32)

    def test_topk_prm_refines_only_selected_candidates(self):
        prm = PointRelationModule(
            d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
            num_classes=2, dropout=0.0, topk=2,
        ).eval()
        prm.transformer = nn.Identity()
        with torch.no_grad():
            prm.offset_head.weight.zero_()
            prm.offset_head.bias.copy_(torch.tensor([1.0, 2.0]))
            prm.class_head.weight.zero_()
            prm.class_head.bias.copy_(torch.tensor([3.0, 4.0]))

        offsets = torch.zeros(1, 6, 2)
        scores = torch.tensor([[[0.0, value] for value in range(6)]])
        anchors = torch.tensor([[[float(i), 0.0] for i in range(6)]])
        refined_offsets, refined_scores = prm(offsets, scores, anchors)

        selected = torch.tensor([4, 5])
        unselected = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.equal(refined_offsets[0, unselected], offsets[0, unselected]))
        self.assertTrue(torch.equal(refined_scores[0, unselected], scores[0, unselected]))
        self.assertTrue(torch.allclose(
            refined_offsets[0, selected], torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        ))
        self.assertTrue(torch.allclose(
            refined_scores[0, selected], scores[0, selected] + torch.tensor([3.0, 4.0])
        ))
        self.assertEqual(prm.last_attention_tokens, 2)

    def test_topk_at_or_above_candidate_count_matches_global_prm(self):
        global_prm = PointRelationModule(
            d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
            num_classes=2, dropout=0.0, topk=0,
        ).eval()
        topk_prm = PointRelationModule(
            d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
            num_classes=2, dropout=0.0, topk=20,
        ).eval()
        topk_prm.load_state_dict(global_prm.state_dict())
        offsets = torch.randn(1, 12, 2)
        scores = torch.randn(1, 12, 2)
        anchors = torch.randn(1, 12, 2)
        with torch.no_grad():
            expected = global_prm(offsets, scores, anchors)
            actual = topk_prm(offsets, scores, anchors)
        self.assertTrue(torch.allclose(expected[0], actual[0]))
        self.assertTrue(torch.allclose(expected[1], actual[1]))

    def test_shapes_and_c3_to_p4_gradient_path(self):
        bifpn = BiFPN(8, 16, 32, feature_size=4)
        c3 = torch.randn(2, 8, 17, 19, requires_grad=True)
        c4 = torch.randn(2, 16, 8, 9, requires_grad=True)
        c5 = torch.randn(2, 32, 4, 4, requires_grad=True)

        p3, p4, p5 = bifpn([c3, c4, c5])

        self.assertEqual(p3.shape, (2, 4, 17, 19))
        self.assertEqual(p4.shape, (2, 4, 8, 9))
        self.assertEqual(p5.shape, (2, 4, 4, 4))

        p4.mean().backward()
        self.assertIsNotNone(c3.grad)
        self.assertGreater(c3.grad.abs().sum().item(), 0.0)

    def test_all_fusion_weights_receive_gradients(self):
        bifpn = BiFPN(8, 16, 32, feature_size=4)
        outputs = bifpn([
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 32, 4, 4),
        ])
        sum(feature.square().mean() for feature in outputs).backward()

        for name, module in bifpn.named_modules():
            if name.endswith('fusion'):
                self.assertIsNotNone(module.weights.grad, name)
                self.assertTrue(torch.isfinite(module.weights.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
