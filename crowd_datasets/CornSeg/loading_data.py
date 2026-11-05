import torchvision.transforms as standard_transforms
from .CornSeg import CornSeg


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(data_root, crop_size=384, train_list='corn_seg_train.list',
                 val_list='corn_seg_val.list', eval_min_size=384, eval_max_size=768):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    train_set = CornSeg(
        data_root, train=True, transform=transform, patch=True, flip=True,
        crop_size=crop_size, list_file=train_list,
        eval_min_size=eval_min_size, eval_max_size=eval_max_size,
    )
    val_set = CornSeg(
        data_root, train=False, transform=transform, list_file=val_list,
        eval_min_size=eval_min_size, eval_max_size=eval_max_size,
    )

    return train_set, val_set
