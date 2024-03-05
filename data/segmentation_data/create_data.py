import sys

sys.path.append("/home/daniel/deepvalve/src/segmentation/")
sys.path.append("/home/daniel/deepvalve/src/")


import argparse
import config as cfg
import data_loader as dl
import utils


def main(
    mask_generation_method=cfg.mask_generation_method,
    base_folder_path=cfg.path_for_segmentation_data_folder,
    no_of_points=cfg.no_of_points,
    mask_thickness=cfg.mask_thickness,
):

    training_loader = dl.get_data_loader(
        mode="train", batch_size=8, num_workers=8, normalize_keypts=False
    )
    val_loader = dl.get_data_loader(
        mode="val", batch_size=8, num_workers=8, normalize_keypts=False
    )
    testing_loader = dl.get_data_loader(
        mode="test", batch_size=8, num_workers=8, normalize_keypts=False
    )

    if mask_generation_method == "polygon":

        if no_of_points is None:
            no_of_points = cfg.no_of_points
        train, val, test = utils.create_dataloaders_with_masks_polygons(
            training_loader, val_loader, testing_loader, no_of_points=cfg.no_of_points
        )

    else:

        if no_of_points is None:
            no_of_points = cfg.no_of_points
        if mask_thickness is None:
            mask_thickness = cfg.mask_thickness
        train, val, test = utils.create_dataloaders_with_masks_line(
            training_loader,
            val_loader,
            testing_loader,
            no_of_points=cfg.no_of_points,
            mask_thickness=cfg.mask_thickness,
        )

    utils.save_images_and_masks_in_folders(
        train,
        val,
        test,
        mask_generation_method,
        base_folder_path=cfg.path_for_segmentation_data_folder,
    )


if __name__ == "__main__":

    main()
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--mask_generation_method', type=str, help='mask generation method', default='polygon')
    # parser.add_argument('--base_folder_path', type=str, help='base folder path', default='../segmentation_data')
    # parser.add_argument('--no_of_points', type=int, help='no of points', default=cfg.no_of_points)
    # parser.add_argument('--mask_thickness', type=int, help='mask thickness', default=cfg.mask_thickness)
    # args = parser.parse_args()
    # main(args.mask_generation_method, args.base_folder_path, args.no_of_points, args.mask_thickness)
