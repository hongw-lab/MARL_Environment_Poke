from run_simulation_xy import run_simulation
import argparse
import os
import ray
import importlib
from ray import tune
from current_config_xy import get_config
from ray.rllib.agents.ppo import PPOTrainer
from MultiAgentSync_fullobs_samefield_randnose_xycoords import (
    MultiAgentSync_fullobs,
    MultiAgentSing_fullobs,
    MultiAgentSync_noobs,
    MultiAgentSing_noobs,
)
import random
from scipy.io import savemat
import numpy as np


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
        "--checkpoint_dir",
        type=str,
        default=None,
        help="full path to all checkpoints",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./rollout_results",
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
    args = parser.parse_args()

    if args.num_gpus > 0 and not torch.cuda.is_available():
        print("No GPU available. Setting num_gpus to 0.")
        args.num_gpus = 0  # Use CPU if no GPU is available

    print("Running trails with the following arguments: ", args)
    return args


def roll_out_events(checkpoint_dir, env, step, config, save_path):
    analysis = tune.ExperimentAnalysis(os.path.abspath(checkpoint_dir))
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean",
    )
    iterations = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="training_iteration",
    )

    checkpoint_idx = list(range(0, len(checkpoints), step))

    # corresponding iteration number
    x = np.array([i[1] for i in iterations])
    x = x[checkpoint_idx]

    npoke1 = [[] for _ in range(len(checkpoint_idx))]
    npoke2 = [[] for _ in range(len(checkpoint_idx))]
    ndrink1 = [[] for _ in range(len(checkpoint_idx))]
    ndrink2 = [[] for _ in range(len(checkpoint_idx))]
    correct1 = [[] for _ in range(len(checkpoint_idx))]
    correct2 = [[] for _ in range(len(checkpoint_idx))]
    nsync = [[] for _ in range(len(checkpoint_idx))]
    reward1 = [[] for _ in range(len(checkpoint_idx))]
    reward2 = [[] for _ in range(len(checkpoint_idx))]
    poketime1 = [[] for _ in range(len(checkpoint_idx))]
    poketime2 = [[] for _ in range(len(checkpoint_idx))]
    for i in range(len(checkpoint_idx)):
        agent = PPOTrainer(config=config)
        agent.restore(checkpoints[checkpoint_idx[i]][0])
        for j in range(50):
            ntrials, _, eve, rew, _, _ = run_simulation(env, agent)
            npoke1[i].append(ntrials["npoke1"])
            npoke2[i].append(ntrials["npoke2"])
            ndrink1[i].append(ntrials["ndrink1"])
            ndrink2[i].append(ntrials["ndrink2"])
            correct1[i].append(ntrials["ncorrect1"])
            correct2[i].append(ntrials["ncorrect2"])
            nsync[i].append(ntrials["ncorrect"])
            reward1[i].append(rew["ag1"])
            reward2[i].append(rew["ag2"])
            poketime1[i].append(eve["poke1"])
            poketime2[i].append(eve["poke2"])
    # shuffle pokes
    nsyncsh = [[] for _ in range(len(checkpoint_idx))]
    for i in range(len(checkpoint_idx)):
        for e in range(len(poketime1[i])):
            # p1 and p2 are two lists of numbers in the range 0 to 199
            p1 = poketime1[i][e]
            p2 = poketime2[i][e]
            if len(p1) == 0 or len(p2) == 0:
                nsyncsh[i].append(0)
                continue
            # Circularly shift p1 by a random step
            shift_step = random.randint(2, 198)
            shifted_p1 = [(x + shift_step) % 200 for x in p1]

            poke_dif = list()
            for x in shifted_p1:
                min_difference = min(abs(x - y) for y in p2)
                poke_dif.append(min_difference)

            # Count how many times the minimum time difference is within 2 steps
            nsyncsh[i].append(sum(1 for diff in poke_dif if diff <= 2))

    savemat(
        save_path,
        {
            "npoke1": npoke1,
            "ndrink1": ndrink1,
            "correct1": correct1,
            "npoke2": npoke2,
            "ndrink2": ndrink2,
            "correct2": correct2,
            "reward1": reward1,
            "reward2": reward2,
            "poketime1": poketime1,
            "poketime2": poketime2,
            "nsync": nsync,
            "nsyncsh": nsyncsh,
            "x": checkpoint_idx,
        },
    )


if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Set up Ray.
    ray.init(log_to_driver=False, num_gpus=1)

    # Fetch experiment configurations
    config, env_config = get_config(args)
    # get env
    ENV_CLASSES = {
        "MultiAgentSync_fullobs": MultiAgentSync_fullobs,
        "MultiAgentSing_fullobs": MultiAgentSing_fullobs,
        "MultiAgentSync_noobs": MultiAgentSync_noobs,
        "MultiAgentSing_noobs": MultiAgentSing_noobs,
    }

    env_class = ENV_CLASSES.get(args.condition)
    env = env_class(config=env_config)

    # define idx of checkpoints for roll out

    save_path = f"{args.results_dir}/{os.path.basename(os.path.normpath(args.checkpoint_dir))}.mat"
    roll_out_events(args.checkpoint_dir, env, args.eval_step, config, save_path)
