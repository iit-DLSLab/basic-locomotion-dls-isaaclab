from __future__ import annotations

from terrain_reconstruction_networks import _parse_args, run_fake_data_smoke_test, train_from_saved_dataset


if __name__ == "__main__":
    args = _parse_args()
    max_lidar_points = args.max_lidar_points if args.max_lidar_points and args.max_lidar_points > 0 else None
    if args.dataset_path:
        train_from_saved_dataset(
            dataset_path=args.dataset_path,
            model_type="pointnet_gru",
            device=args.device,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            num_epochs=args.epochs if args.epochs is not None else 50,
            validation_fraction=args.validation_fraction,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_lidar_points=max_lidar_points,
            model_path=args.model_path,
            comparison_png_path=args.comparison_png_path,
            comparison_samples=args.comparison_samples,
        )
    else:
        run_fake_data_smoke_test(
            model_type="pointnet_gru",
            device=args.device,
            num_samples=args.num_samples,
            batch_size=args.batch_size if args.batch_size is not None else 8,
            num_epochs=args.epochs if args.epochs is not None else 3,
            max_lidar_points=max_lidar_points,
        )
