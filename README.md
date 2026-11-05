# P2PNet-BiFPN-PRM

This project implements an improved P2PNet model for point-annotated object
counting. It uses VGG16-BN for multi-scale feature extraction, replaces the
original one-way FPN with a single-layer BiFPN, and applies a Point Relation
Module (PRM) to model relationships between candidate points and refine their
predictions.

This directory is a standalone, code-only package. It does not include any
dataset, split file, pretrained model, training checkpoint, log, or experiment
result.

## Model Architecture

The main data flow is:

```text
Input image
    │
    ▼
VGG16-BN backbone
    │
    ├── C3
    ├── C4
    └── C5
         │
         ▼
Single-layer bidirectional weighted BiFPN
         │
         ▼
Classification head + point-offset regression head
         │
         ▼
PRM self-attention refinement
         │
         ▼
Confidence scores + point coordinates
```

The main components are:

- **BiFPN:** performs top-down and bottom-up fusion of C3, C4, and C5 using
  learnable normalized fusion weights.
- **Depthwise separable convolutions:** reduce the cost of feature processing
  after each BiFPN fusion node.
- **PRM:** uses a Transformer encoder to model relationships between candidate
  points and predicts residual corrections for classification logits and point
  offsets.
- **Top-k PRM:** optionally sends only the k highest-confidence candidates
  through PRM to reduce the memory and computation cost of global
  self-attention. Unselected candidates remain in the final prediction set.
- **One-to-one assignment:** uses Hungarian matching to pair predictions with
  ground-truth points during training.
- **Standard losses:** the current BiFPN+PRM configuration uses cross-entropy
  classification loss and MSE point-regression loss.

## Project Structure

```text
P2PNet_BiFPN_PRM_code/
├── crowd_datasets/
│   └── CornSeg/               # CornSeg loading and augmentation code
├── models/
│   ├── backbone.py            # VGG16/VGG16-BN backbone
│   ├── matcher.py             # Hungarian matcher
│   ├── p2pnet.py              # P2PNet, PRM, prediction heads, and losses
│   ├── p2pnet_bifpn.py        # BiFPN and BiFPN+PRM model composition
│   ├── p2pnet_old.py          # Legacy/non-PRM comparison model
│   └── vgg_.py                # VGG network definitions
├── tests/
│   └── test_bifpn.py          # BiFPN, PRM, and Top-k PRM tests
├── util/
│   └── misc.py                # Tensor, logging, and distributed utilities
├── engine.py                  # Training and validation loops
├── infer_test_image.py        # Single-image inference and visualization
├── train.py                   # Shared training program
├── train_bifpn_prm.py         # BiFPN+PRM training entry point
├── train_bifpn_prm_topk.py    # Top-k PRM training entry point
└── requirements.txt           # Version-pinned Python dependencies
```

The active model class is `P2PNetBiFPNPRM`. It inherits from `P2PNet`, enables
PRM, and replaces the original FPN with BiFPN. `p2pnet_old.py` is retained
primarily for the non-PRM BiFPN comparison model.

## Environment

Validated environment:

- Windows
- Python 3.11.15
- PyTorch 2.8.0+cu129
- Torchvision 0.23.0+cu129
- PyTorch CUDA runtime 12.9
- A CUDA-capable NVIDIA GPU (the training program currently requires CUDA)

Create the Conda environment and install the dependencies:

```powershell
conda create -n p2pnet_bifpn_prm python=3.11.15 -y
conda activate p2pnet_bifpn_prm
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes the official PyTorch package index for CUDA 12.9.
When VGG16-BN is constructed for the first time, Torchvision may download its
pretrained weights if they are not already cached locally.

Verify PyTorch and CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Dataset Format

The dataset is not included. Specify its location with `--data_root`. The
recommended layout is:

```text
CornDataset/
├── images/
│   ├── image_001.jpg
│   └── image_002.jpg
├── labels/
│   ├── image_001.txt
│   └── image_002.txt
├── corn_seg_train.list
├── corn_seg_val.list
└── corn_seg_test.list
```

Each line in a `.list` file contains an image path and its point-annotation
path, both relative to the dataset root:

```text
images/image_001.jpg labels/image_001.txt
images/image_002.jpg labels/image_002.txt
```

Each annotation file contains one target-center point per line in `x y`
format:

```text
125.5 86.0
240.0 173.5
```

The origin is at the top-left corner of the image. The x coordinate increases
to the right, and the y coordinate increases downward.

Training uses random scaling, a 384×384 random crop, and random horizontal
flipping by default. Validation and test images are resized while preserving
their aspect ratio. The default target size for the shorter side is 384, and
the maximum size of the longer side is 768.

## Train BiFPN+PRM

Run the following command from the project root:

```powershell
python train_bifpn_prm.py `
  --data_root "D:\datasets\CornDataset" `
  --train_list corn_seg_train.list `
  --val_list corn_seg_val.list `
  --gpu_id 0 `
  --batch_size 4 `
  --epochs 800 `
  --seed 42
```

Common options:

| Option | Default | Description |
|---|---:|---|
| `--lr` | `1e-4` | Learning rate for non-backbone parameters |
| `--lr_backbone` | `1e-5` | Backbone learning rate |
| `--batch_size` | `4` | Training batch size |
| `--epochs` | `800` | Number of training epochs |
| `--lr_drop` | `400` | StepLR decay epoch |
| `--crop_size` | `384` | Random training crop size |
| `--eval_freq` | `5` | Validation interval in epochs |
| `--threshold` | `0.5` | Foreground confidence threshold for validation |
| `--seed` | `42` | Random seed |
| `--num_workers` | `4` | Number of DataLoader workers |
| `--prm_topk` | `0` | `0` sends all candidates through PRM |

The default output layout is:

```text
training_results/maize_counting_452/P2PNet_BiFPN_PRM/seed_<seed>/
├── ckpt/
│   ├── best_val_mae.pth       # Checkpoint with the best validation MAE
│   └── latest.pth             # Latest complete training state
├── log/
│   └── run_log.txt
└── tb/                         # TensorBoard logs
```

Use `--output_dir`, `--checkpoints_dir`, and `--tensorboard_dir` to override
the default locations.

### Resume Training

Automatically resume from `latest.pth` in the default output directory:

```powershell
python train_bifpn_prm.py `
  --data_root "D:\datasets\CornDataset" `
  --seed 42 `
  --auto_resume
```

Alternatively, specify a training-state checkpoint explicitly:

```powershell
python train_bifpn_prm.py `
  --data_root "D:\datasets\CornDataset" `
  --resume "D:\checkpoints\latest.pth"
```

Use `latest.pth` for a complete resume. `best_val_mae.pth` is intended mainly
for inference and evaluation and does not contain the complete optimizer and
learning-rate scheduler state.

## Train Top-k PRM

For example, send only the 256 highest-confidence candidates through PRM:

```powershell
python train_bifpn_prm_topk.py `
  --data_root "D:\datasets\CornDataset" `
  --prm_topk 256 `
  --gpu_id 0 `
  --batch_size 4 `
  --epochs 800 `
  --seed 42
```

`train_bifpn_prm_topk.py` requires `--prm_topk` to be a positive integer.
Candidates are ranked using their pre-PRM foreground probabilities. Candidates
outside the top-k bypass PRM refinement but still participate in the final
confidence filtering step.

## Single-Image Inference

`infer_test_image.py` locates a requested image through
`corn_seg_test.list` and writes the predicted points, count summary, and
visualization:

```powershell
python infer_test_image.py `
  --data-root "D:\datasets\CornDataset" `
  --image-name image_001.jpg `
  --weight-path "training_results\maize_counting_452\P2PNet_BiFPN_PRM\seed_42\ckpt\best_val_mae.pth" `
  --output-dir "inference_results\image_001" `
  --threshold 0.5 `
  --gpu-id 0
```

The output directory contains:

- `*_inference_comparison.png`: input, ground-truth points, and predicted
  points.
- `*_prediction_points.csv`: predicted coordinates, confidence scores, and
  coordinate-validity flags.
- `*_inference_summary.json`: ground-truth count, predicted count, absolute
  error, and inference settings.

The inference program falls back to CPU when CUDA is unavailable.

## TensorBoard

```powershell
tensorboard --logdir training_results\maize_counting_452\P2PNet_BiFPN_PRM\seed_42\tb
```

Open the local URL printed by TensorBoard.

## Tests

```powershell
python -B -m unittest discover -s tests -p test_bifpn.py -v
```

The tests cover:

- BiFPN output shapes.
- The C3-to-P4 gradient path.
- Gradient propagation to every BiFPN fusion weight.
- BiFPN+PRM forward and backward passes.
- Refinement of only the selected candidates in Top-k PRM.
- Equivalence between Top-k PRM and global PRM when k is at least the number
  of candidates.

## Notes

- This package does not include a dataset or trained weights.
- The training entry points always enable BiFPN and PRM and disable focal loss
  and Smooth L1 loss.
- The training program currently uses `torch.device('cuda')`, so training
  requires a working CUDA GPU.
- The confidence threshold directly affects the predicted count. Select it on
  the validation set only; do not tune it on the test set.
- Different `--seed` values use separate output directories by default, which
  makes multi-seed experiments easier to manage.
