"""Train a BiFPN+PRM model whose attention is restricted to top-k candidates.

Candidate ranking uses the pre-PRM foreground probability.  Only selected
candidates enter self-attention; all candidates remain in the final detector.
Outputs are isolated by k and random seed so runs can be resumed safely.
"""

import argparse
from pathlib import Path

import train


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, parents=[train.get_args_parser()]
    )
    parser.add_argument(
        '--auto_resume', action='store_true',
        help='resume checkpoints_dir/latest.pth if it exists',
    )
    args = parser.parse_args()
    if args.prm_topk <= 0:
        parser.error('--prm_topk must be a positive integer for top-k training')
    if args.use_p3 or args.use_focal_loss or args.use_smooth_l1:
        parser.error('BiFPN+top-k PRM uses P4 prediction with standard CE/MSE losses')

    args.use_bifpn = True
    args.use_prm = True
    args.use_p3 = False
    args.use_focal_loss = False
    args.use_smooth_l1 = False

    project_root = Path(__file__).resolve().parent
    seed_root = (
        project_root / 'training_results' / 'maize_counting_452'
        / 'reviewer_topk' / f'P2PNet_BiFPN_PRM_topk{args.prm_topk}'
        / f'seed_{args.seed}'
    )
    if args.output_dir == './log':
        args.output_dir = str(seed_root / 'log')
    if args.checkpoints_dir == './ckpt':
        args.checkpoints_dir = str(seed_root / 'ckpt')
    if args.tensorboard_dir == './runs':
        args.tensorboard_dir = str(seed_root / 'tb')

    latest = Path(args.checkpoints_dir) / 'latest.pth'
    if args.auto_resume and args.resume:
        parser.error('use either --auto_resume or --resume, not both')
    if args.auto_resume and latest.is_file():
        args.resume = str(latest)
        print(f'Auto-resuming from {latest}')
    elif not args.resume and latest.is_file():
        parser.error(
            f'{latest} already exists; use --auto_resume, --resume, or a '
            'different --checkpoints_dir'
        )
    return args


if __name__ == '__main__':
    train.main(parse_args())
