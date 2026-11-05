import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr_drop', default=400, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.005, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    # * Focal Loss
    parser.add_argument('--use_focal_loss', action='store_true',
                        help="Use focal loss instead of standard cross-entropy")
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help="Alpha parameter for focal loss")
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help="Gamma parameter for focal loss")

    # * Smooth L1 Loss
    parser.add_argument('--use_smooth_l1', action='store_true',
                        help="Use smooth L1 loss instead of MSE for point regression")
    parser.add_argument('--smooth_l1_beta', default=1.0, type=float,
                        help="Beta parameter for smooth L1 loss")

    # * LR Schedule
    parser.add_argument('--use_cosine_lr', action='store_true',
                        help="Use cosine annealing LR schedule instead of step LR")
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help="Number of warmup epochs with linear LR increase")

    # * Crop size
    parser.add_argument('--crop_size', default=384, type=int,
                        help="Random crop size for training patches")

    # * Point Relation Module (PRM)
    parser.add_argument('--use_prm', action='store_true',
                        help="Use Point Relation Module for neighbor-aware refinement")
    parser.add_argument('--prm_d_model', default=64, type=int,
                        help="Hidden dimension of PRM self-attention")
    parser.add_argument('--prm_nhead', default=4, type=int,
                        help="Number of attention heads in PRM")
    parser.add_argument('--prm_num_layers', default=2, type=int,
                        help="Number of self-attention layers in PRM")
    parser.add_argument('--prm_dim_feedforward', default=128, type=int,
                        help="FFN dimension in PRM")
    parser.add_argument(
        '--prm_topk', default=0, type=int,
        help='number of highest-confidence candidates refined by PRM; 0 uses global attention',
    )

    parser.add_argument('--use_old_inc_model', action='store_true', help='Use the incremental old-model architecture for training')
    parser.add_argument('--use_bifpn', action='store_true',
                        help='Use the original P2PNet with its FPN replaced by BiFPN')
    parser.add_argument('--use_p3', action='store_true',
                        help='Use the original P2PNet with P3 prediction and 1x1 anchors')
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='CornSeg')
    parser.add_argument('--data_root', default='./crowd_datasets/Corn_datasets_merged_452',
                        help='path where the dataset is')
    parser.add_argument('--train_list', default='corn_seg_train.list')
    parser.add_argument('--val_list', default='corn_seg_val.list')
    parser.add_argument('--test_list', default='corn_seg_test.list')
    parser.add_argument('--eval_min_size', default=384, type=int)
    parser.add_argument('--eval_max_size', default=768, type=int)
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='validation confidence threshold; tune on val only')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)

    return parser

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f'--threshold must be in [0, 1], got {args.threshold}')
    if args.prm_topk < 0:
        raise ValueError(f'--prm_topk must be non-negative, got {args.prm_topk}')
    # create output directories if not exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    # Preserve the existing log when continuing from latest.pth.
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    log_mode = 'a' if args.resume else 'w'
    with open(run_log_name, log_mode) as log_file:
        if args.resume:
            log_file.write('Resume Log %s from %s\n' % (time.strftime("%c"), args.resume))
        else:
            log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(
        param_dicts,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    # LR scheduler: cosine annealing or step LR
    if getattr(args, 'use_cosine_lr', False):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    data_generator = torch.Generator()
    data_generator.manual_seed(seed)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers,
                                   worker_init_fn=seed_worker, generator=data_generator)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd,
                                    num_workers=args.num_workers, worker_init_fn=seed_worker,
                                    generator=data_generator)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        # latest.pth is created by this project and includes optimizer plus
        # NumPy/Python RNG states. PyTorch >=2.6 otherwise defaults to the
        # restricted weights-only loader, which cannot deserialize those states.
        checkpoint = torch.load(
            args.resume, map_location='cpu', weights_only=False
        )
        if not args.eval:
            required = {'model', 'optimizer', 'lr_scheduler', 'epoch'}
            missing = sorted(required.difference(checkpoint))
            if missing:
                raise ValueError(
                    f'Resume checkpoint is missing {missing}. Use latest.pth, not best_val_mae.pth.'
                )
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint.get('best_val_mae', best_val_mae)
        best_val_rmse = checkpoint.get('best_val_rmse', best_val_rmse)
        if 'data_generator_state' in checkpoint:
            data_generator.set_state(checkpoint['data_generator_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_states' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    rmse = []
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    
    # warmup: linearly increase LR from 0 to target LR over warmup_epochs
    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        # warmup: scale LR linearly
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base_lr * warmup_factor

        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))
                
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler (skip during warmup)
        if warmup_epochs == 0 or epoch >= warmup_epochs:
            lr_scheduler.step()
        # run evaluation
        if (epoch + 1) % args.eval_freq == 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(
                model, data_loader_val, device, threshold=args.threshold
            )
            t2 = time.time()

            mae.append(result[0])
            rmse.append(result[1])
            # print the evaluation results
            print('=======================================test=======================================')
            print("mae:", result[0], "rmse:", result[1], "time:", t2 - t1,
                  "best mae:", min(best_val_mae, result[0]))
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, rmse:{}, time:{}, best mae:{}".format(
                    result[0], result[1], t2 - t1, min(best_val_mae, result[0])))
            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/rmse@{}: {}".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/rmse', result[1], step)
                step += 1

            is_better = (result[0] < best_val_mae) or (
                result[0] == best_val_mae and result[1] < best_val_rmse
            )
            if is_better:
                best_val_mae, best_val_rmse = result
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_val_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    'best_val_mae': best_val_mae,
                    'best_val_rmse': best_val_rmse,
                    'args': vars(args),
                }, checkpoint_best_path)

        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_val_mae': best_val_mae,
            'best_val_rmse': best_val_rmse,
            'args': vars(args),
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all(),
            'data_generator_state': data_generator.get_state(),
        }, checkpoint_latest_path)
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
