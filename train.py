import argparse
import os
import ray
import importlib
from ray import tune
from current_config_xy import get_config
from ray.rllib.agents.ppo import PPOTrainer
import re


def get_args():
    import torch

    parser = argparse.ArgumentParser(description="Train MARL cooperation agents")

    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="which gpu to use - 0 or 1",
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.3,
        help="Number of GPUs to run on (can be a fraction)",
    )
    parser.add_argument(
        "--env_dir",
        type=str,
        default="MultiAgentSync_fullobs_samefield_randnose_xycoords.py",
        help="environment file",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=[
            "MultiAgentSync_fullobs",
            "MultiAgentSing_fullobs",
            "MultiAgentSync_noobs",
            "MultiAgentSing_noobs",
        ],
        default="MultiAgentSync_fullobs",
        help="condition choices: non-coop/coop, fullobs/noobs",
    )

    parser.add_argument(
        "--load_policy",
        type=str,
        default=None,
        help="directory to the previously-trained policy to load",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--l2_curr", type=float, default=0.01, help="L2-reg on recurrence"
    )

    parser.add_argument("--l2_inp", type=float, default=0, help="L2-reg on input")
    parser.add_argument(
        "--coop_window", type=int, default=2, help="time window for agents cooperation"
    )
    parser.add_argument(
        "--randomize_loc",
        type=bool,
        default=True,
        help="randomize np and water after correct, generally true",
    )
    parser.add_argument(
        "--randomize_miss",
        type=bool,
        default=False,
        help="randomize np and water after miss, generally false",
    )
    parser.add_argument(
        "--train_iter", type=int, default=4000, help="number of training iterations"
    )
    parser.add_argument("--checkpoint_frequency", type=int, default=10)
    parser.add_argument("--resume", default=False)

    args = parser.parse_args()

    if args.num_gpus > 0 and not torch.cuda.is_available():
        print("No GPU available. Setting num_gpus to 0.")
        args.num_gpus = 0  # Use CPU if no GPU is available

    print("Running trails with the following arguments: ", args)
    return args


if __name__ == "__main__":

    args = get_args()
    if not args.load_policy:
        restore_path = None
    else:
        analysis = tune.ExperimentAnalysis(os.path.abspath(args.load_policy))
        restore_path = analysis.get_best_checkpoint(
            trial=analysis.get_best_trial(metric="episode_reward_mean"),
            metric="episode_reward_mean",
            mode="max",
            return_path=True,
        )
        print(restore_path)
        last_folder = os.path.basename(restore_path)
        checkpoint_number = int(last_folder[11:])
        args.train_iter += checkpoint_number
    print(f"restoring{restore_path}, target iterations {args.train_iter}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Set up Ray.
    ray.init(log_to_driver=False, num_gpus=1)

    # Fetch experiment configurations
    config, env_config = get_config(args)

    # Create progress report
    from ray.tune import CLIReporter

    reporter = CLIReporter(
        max_progress_rows=10, max_report_frequency=300, infer_limit=8
    )
    reporter.add_metric_column("policy_reward_mean")

    # Run training
    analysis = tune.run(
        PPOTrainer,
        config=config,
        progress_reporter=reporter,
        checkpoint_freq=args.checkpoint_frequency,
        checkpoint_at_end=True,
        stop={
            "training_iteration": args.train_iter,
        },
        local_dir=args.output_dir,
        verbose=1,
        restore=restore_path,
        resume=True if args.resume == "True" else False,
    )

    # Find best result - the dir here can be used for the next training stage
    best_result = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial(metric="episode_reward_mean"),
        metric="episode_reward_mean",
        mode="max",
    )
    print(f"best iteration{best_result}")

    ray.shutdown()
