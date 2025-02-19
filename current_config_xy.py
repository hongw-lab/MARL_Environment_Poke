from MultiAgentSync_fullobs_samefield_randnose_xycoords import (
    MultiAgentSync_fullobs,
    MultiAgentSing_fullobs,
    MultiAgentSync_noobs,
    MultiAgentSing_noobs,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTorchPolicy
from Customcallback import CustomCallbacks
from simple_rnn_v2_3_2 import AnotherTorchRNNModel
import torch


def get_config(args):
    env_config = {
        "height": 8,
        "width": 8,
        "sync_limit": args.coop_window,
        "randomize": args.randomize_loc,
        "randomize_miss": args.randomize_miss,
    }
    ENV_CLASSES = {
        "MultiAgentSync_fullobs": MultiAgentSync_fullobs,
        "MultiAgentSing_fullobs": MultiAgentSing_fullobs,
        "MultiAgentSync_noobs": MultiAgentSync_noobs,
        "MultiAgentSing_noobs": MultiAgentSing_noobs,
    }

    env_class = ENV_CLASSES.get(args.condition)
    if env_class:
        env = env_class(config=env_config)
    else:
        raise ValueError(f"Environment '{args.condition}' not found.")

    ModelCatalog.register_custom_model("rnn2", AnotherTorchRNNModel)

    policies = {
        "policy1": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
        "policy2": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
    }

    # 2) Defines an agent->policy mapping function.
    def policy_mapping_fn(agent_id: str) -> str:
        # Make sure agent ID is valid.
        assert agent_id in ["agent1", "agent2"], f"ERROR: invalid agent ID {agent_id}!"
        ### Modify Code here ####
        id = agent_id[-1]
        return f"policy{id}"

    config = {
        "env": env_class,
        "env_config": env_config,
        "num_workers": 0,
        "exploration_config": {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            "eta": 1.5,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 64,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "use_lstm": False,
            },
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            "sub_exploration": {
                "type": "StochasticSampling",
            },
        },
        # !PyTorch users!
        "framework": "torch",  # If users have chosen to install torch instead of tf.
        "create_env_on_driver": True,
    }

    max_seq_len = 10

    # 3) RNN config.
    config.update(
        {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "model": {
                "custom_model": "rnn2",
                "max_seq_len": max_seq_len,
                "custom_model_config": {
                    "rnn_hidden_size": 256,
                    "l2_lambda": args.l2_curr,
                    "l2_lambda_inp": args.l2_inp,
                    "noise_std": 0,
                    "device": torch.device("cuda:0"),
                },
            },
            "num_workers": 0,
            "num_gpus": args.num_gpus,
            "callbacks": CustomCallbacks,
        }
    )

    print()
    print(f"agent1 is now mapped to {policy_mapping_fn('agent1')}")
    print(f"agent2 is now mapped to {policy_mapping_fn('agent2')}")
    return config, env_config
