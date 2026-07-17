import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Process Reward Optimization.")

    # update paths here!
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/shared-local/aoq951/HF_CACHE/",
        help="HF cache directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/shared-local/aoq951/ReNO/outputs",
        help="Directory to save images",
    )

    # model and optimizer
    parser.add_argument("--model", type=str, default="sdxl-turbo")
    parser.add_argument("--lr", type=float, default=5.0)
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--n_inference_steps", type=int, default=1)

    parser.add_argument(
        "--optim",
        choices=["sgd", "adam", "lbfgs"],
        default="sgd",
    )

    parser.add_argument("--nesterov", default=True, action="store_false")
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    # reward losses
    parser.add_argument("--disable_hps",
                        default=True,
                        action="store_false",
                        dest="enable_hps")
    parser.add_argument("--hps_weighting", type=float, default=5.0)

    parser.add_argument("--disable_imagereward",
                        default=True,
                        action="store_false",
                        dest="enable_imagereward")
    parser.add_argument("--imagereward_weighting", type=float, default=1.0)

    parser.add_argument("--disable_clip",
                        default=True,
                        action="store_false",
                        dest="enable_clip")
    parser.add_argument("--clip_weighting", type=float, default=0.01)

    parser.add_argument("--disable_pickscore",
                        default=True,
                        action="store_false",
                        dest="enable_pickscore")
    parser.add_argument("--pickscore_weighting", type=float, default=0.05)

    parser.add_argument("--disable_aesthetic",
                        default=False,
                        action="store_false",
                        dest="enable_aesthetic")
    parser.add_argument("--aesthetic_weighting", type=float, default=0.0)

    parser.add_argument("--disable_reg",
                        default=True,
                        action="store_false",
                        dest="enable_reg")
    parser.add_argument("--reg_weight", type=float, default=0.01)

    # fairness: gradient-based reward loss that pushes the FairFace race
    # classifier's predicted distribution on the in-progress image toward a
    # target distribution (uniform by default). See rewards/fairness.py.
    parser.add_argument("--enable_fairness", action="store_true", default=False)
    parser.add_argument("--fairness_weighting", type=float, default=1.0)
    parser.add_argument(
        "--fairface_weights",
        type=str,
        default="fairface_weights/res34_fair_align_multi_7_20190809.pt",
    )
    parser.add_argument(
        "--fairness_target_dist",
        type=float,
        nargs=7,
        default=None,
        help="Target probability for each of [White, Black, Latino_Hispanic, "
        "East Asian, Southeast Asian, Indian, Middle Eastern]; must sum to "
        "1. Defaults to uniform.",
    )

    # bias correction: counteracts the systematic latent-space drift induced
    # by reward optimization (see training/bias_correction.py)
    parser.add_argument("--enable_debias", action="store_true", default=False)
    parser.add_argument(
        "--debias_direction_path",
        type=str,
        default="bias_directions/default.pt",
        help="Path to a bias direction vector produced by scripts/estimate_bias_direction.py",
    )
    parser.add_argument(
        "--debias_scale",
        type=float,
        default=0.5,
        help="Fraction of the accumulated drift along the bias direction to "
        "remove each iteration (0=off/pure ReNO, 1=fully cancel that "
        "component every step, >1=overcorrect in the opposite direction)",
    )

    # task
    parser.add_argument(
        "--task",
        type=str,
        default="single",
        choices=[
            "t2i-compbench",
            "single",
            "parti-prompts",
            "geneval",
            "example-prompts",
        ],
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A red dog and a green cat",
    )

    parser.add_argument(
        "--benchmark_reward",
        default="total",
        choices=["ImageReward", "PickScore", "HPS", "CLIP", "total"],
    )

    # general
    parser.add_argument("--save_all_images", action="store_true", default=False)
    parser.add_argument("--no_optim", action="store_true", default=False)
    parser.add_argument("--imageselect", action="store_true", default=False)
    parser.add_argument("--memsave", action="store_true", default=False)

    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device_id", type=str, default=None)

    parser.add_argument(
        "--cpu_offloading",
        action="store_true",
        default=False,
    )

    # optional multi-step model
    parser.add_argument("--enable_multi_apply", action="store_true", default=False)
    parser.add_argument("--multi_step_model", type=str, default="flux")

    # Ignore unknown arguments (e.g. Jupyter's -f kernel.json)
    parsed_args, _ = parser.parse_known_args(args)

    return parsed_args