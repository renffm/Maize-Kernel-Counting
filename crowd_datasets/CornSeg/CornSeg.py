import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2


def resize_tensor_and_points(img, points, new_h, new_w):
    """Resize to an explicit integer size and keep point coordinates exact."""
    old_h, old_w = img.shape[-2:]
    new_h = max(1, int(new_h))
    new_w = max(1, int(new_w))
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
    ).squeeze(0)
    if len(points) > 0:
        points[:, 0] *= new_w / old_w
        points[:, 1] *= new_h / old_h
    return img, points


class CornSeg(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False,
                 crop_size=384, list_file=None, eval_min_size=384, eval_max_size=768):
        self.root_path = data_root
        self.crop_size = crop_size
        self.eval_min_size = eval_min_size
        self.eval_max_size = eval_max_size
        default_list = "corn_seg_train.list" if train else "corn_seg_val.list"
        self.img_list_file = (list_file or default_list).split(',')

        self.img_map = {}
        self.img_list = []
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        img, point = load_data((img_path, gt_path), self.train)
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            scale = random.uniform(*scale_range)
            cur_h, cur_w = img.shape[-2:]
            new_h = max(1, int(round(cur_h * scale)))
            new_w = max(1, int(round(cur_w * scale)))
            img, point = resize_tensor_and_points(img, point, new_h, new_w)
            # Ensure both sides are large enough for the fixed crop.
            cur_h, cur_w = img.shape[-2:]
            if cur_h < self.crop_size or cur_w < self.crop_size:
                target_h = max(self.crop_size, cur_h)
                target_w = max(self.crop_size, cur_w)
                img, point = resize_tensor_and_points(
                    img, point, target_h, target_w
                )
        else:
            cur_h, cur_w = img.shape[-2:]
            scale = min(self.eval_min_size / min(cur_h, cur_w),
                        self.eval_max_size / max(cur_h, cur_w))
            new_h = max(1, int(round(cur_h * scale)))
            new_w = max(1, int(round(cur_w * scale)))
            img, point = resize_tensor_and_points(img, point, new_h, new_w)
        if self.train and self.patch:
            img, point = random_crop(img, point, crop_size=self.crop_size)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = (self.crop_size - 1) - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = torch.tensor([index], dtype=torch.long)
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
            if not self.train:
                target[i]['name'] = os.path.basename(img_path)

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            line = line.strip()
            if len(line) < 2:
                continue
            x = float(line.split(' ')[0])
            y = float(line.split(' ')[1])
            points.append([x, y])

    return img, np.asarray(points, dtype=np.float32).reshape(-1, 2)


def random_crop(img, den, num_patch=1, crop_size=384):
    half_h = crop_size
    half_w = crop_size
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    max_start_h = img.size(1) - half_h
    max_start_w = img.size(2) - half_w
    if max_start_h < 0 or max_start_w < 0:
        raise ValueError(
            f'Cannot crop {crop_size}x{crop_size} from image '
            f'{img.size(1)}x{img.size(2)}'
        )
    for i in range(num_patch):
        start_h = random.randint(0, max_start_h)
        start_w = random.randint(0, max_start_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        if len(den) > 0:
            idx = (den[:, 0] >= start_w) & (den[:, 0] < end_w) & (den[:, 1] >= start_h) & (den[:, 1] < end_h)
            record_den = den[idx].copy()
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
        else:
            record_den = den.copy()
        result_den.append(record_den)

    return result_img, result_den
