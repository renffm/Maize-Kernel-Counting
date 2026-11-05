"""Dataset-loader code for the standalone BiFPN+PRM package.

The actual dataset, split lists, images, and annotations are intentionally not
included. Pass their location with ``--data_root``.
"""


def build_dataset(args):
    if args.dataset_file != "CornSeg":
        raise ValueError(
            f"This code package only includes the CornSeg loader, got: "
            f"{args.dataset_file!r}"
        )

    from crowd_datasets.CornSeg.loading_data import loading_data

    crop_size = getattr(args, "crop_size", 384)

    def loader(data_root):
        return loading_data(
            data_root,
            crop_size=crop_size,
            train_list=getattr(args, "train_list", "corn_seg_train.list"),
            val_list=getattr(args, "val_list", "corn_seg_val.list"),
            eval_min_size=getattr(args, "eval_min_size", 384),
            eval_max_size=getattr(args, "eval_max_size", 768),
        )

    return loader
