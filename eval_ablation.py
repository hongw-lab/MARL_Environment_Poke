# ablate certain neurons and test its effect on agent behavior
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
from scipy.io import savemat, loadmat 
import copy
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
        "--ablation_list_dir",
        type=str,
        help="file that contains the list of neurons to ablate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./ablation_results",
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

def find_best_iteration(ck_path):
    ck_path = os.path.abspath(ck_path)
    # print(f"Received path: {ck_path}")
    analysis = tune.ExperimentAnalysis(ck_path)
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean",
    )
    # checkpoints = checkpoints[:400]
    rewards_learn = list()
    for j in range(len(checkpoints)):
        rewards_learn.append(checkpoints[j][1])
    max_number = np.max(rewards_learn)
    idx = np.where(rewards_learn == max_number)[0][0]
    checkpoint_dir = checkpoints[idx][0]
    print(f"best iteration {(idx+1)*10} rewards {max_number}")
    return checkpoint_dir


def run_env(env,agent,nrollout):
    rew = list()
    ncor = list()
    nmiss = list()
    for r in range(nrollout):
        obs = env.reset()
        state1 = [np.zeros(256, np.float32) for _ in range(2)]
        state2 = [np.zeros(256, np.float32) for _ in range(2)]
        while True:
            a1, state1, _ = agent.compute_single_action(
                observation=obs["agent1"], state=state1, policy_id="policy1"
            )
            #
            a2, state2, _ = agent.compute_single_action(
                observation=obs["agent2"], state=state2, policy_id="policy2"
            )
            obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})
            env.render_image()
            if dones["agent1"]:
                break
        rew.append(env.agent1_R + env.agent2_R)
        ncor.append(env.ncorrect)
        nmiss.append(env.nmiss1 + env.nmiss2)
    return rew, env, ncor, nmiss

def ablate_neurons(agent, env, config, ablate_list, save_path):
    # ablate_list["agent1"][0] = [...];
    max_ablation = max((len(v) for v in ablate_list['agent1']), default=0)
    n_pop = len(ablate_list['agent1']) 
    rew = np.full((n_pop, max_ablation + 1), np.nan)  #  population, n_removed
    ncor = np.full((n_pop, max_ablation + 1), np.nan)
    nmiss = np.full((n_pop, max_ablation + 1), np.nan)
    weights = agent.get_weights()
    for grp in range(n_pop):
        n_sil_range = range(len(ablate_list['agent1'][grp]) + 1) # from ablate 0 to ablate all on the list 
        for n_sil, rm in enumerate(n_sil_range):
            silence = ablate_list['agent1'][grp][:n_sil]  # important note: matlab index to python index, -1 if necessary
            pol1 = copy.deepcopy(weights["policy1"])
            pol1["action_branch.weight"][:, silence] = 0
            pol1["value_branch.weight"][:, silence] = 0
            pol1["rnn.weight_hh_l0"][:, silence] = 0
            pol1["rnn.weight_hh_l0"][silence, :] = 0
            silence = ablate_list['agent2'][grp][:n_sil] 
            pol2 = copy.deepcopy(weights["policy2"])
            pol2["action_branch.weight"][:, silence] = 0
            pol2["value_branch.weight"][:, silence] = 0
            pol2["rnn.weight_hh_l0"][:, silence] = 0
            pol2["rnn.weight_hh_l0"][silence, :] = 0
            agent_ablated = PPOTrainer(config=config)
            agent_ablated.set_weights(
                {
                    "policy1": pol1,
                    "policy2": pol2,
                }
            )
            rew_, env, ncor_, nmiss_ = run_env(env, agent_ablated, 10)
            rew[grp, rm] = sum(rew_) / len(rew_)
            ncor[grp, rm] = sum(ncor_) / len(ncor_)
            nmiss[grp, rm] = sum(nmiss_) / len(nmiss_)
    savemat(
        save_path,
        {
            "rew": rew,
            "ncor": ncor,
            "nmiss": nmiss,
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

    # Find best agent
    checkpoint_dir = find_best_iteration(args.checkpoint_dir)
    agent = PPOTrainer(config=config)
    agent.restore(checkpoint_dir)

    # ablate list 

    # -- fake example for structure 
    # ablate_list = {"agent1":[[1,5,3,6],[188,29]],"agent2":[[1,5,3,6],[188,29,84]]}
    ablate_list = {key: value for key, value in loadmat(args.ablation_list_dir).items() if not key.startswith("__")}

    
    save_path = f"{args.results_dir}/{os.path.basename(os.path.normpath(args.checkpoint_dir))}.mat"
    ablate_neurons(agent, env, config, ablate_list, save_path)
