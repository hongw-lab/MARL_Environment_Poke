from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):

        episode.custom_metrics["nmiss1"] = episode._agent_to_last_info["agent1"][
            "nmiss"
        ]
        episode.custom_metrics["nmiss2"] = episode._agent_to_last_info["agent2"][
            "nmiss"
        ]
        episode.custom_metrics["ncorrect"] = episode._agent_to_last_info["agent1"][
            "ncorrect"
        ]  # number of synchronized pokes
        episode.custom_metrics["npoke1"] = episode._agent_to_last_info["agent1"][
            "npoke1"
        ]
        episode.custom_metrics["npoke2"] = episode._agent_to_last_info["agent2"][
            "npoke2"
        ]
        episode.custom_metrics["ndrink1"] = episode._agent_to_last_info["agent1"][
            "ndrink1"
        ]
        episode.custom_metrics["ndrink2"] = episode._agent_to_last_info["agent2"][
            "ndrink2"
        ]
        episode.custom_metrics["ncorrect1"] = episode._agent_to_last_info["agent1"][
            "ncorrect1"
        ]
        episode.custom_metrics["ncorrect2"] = episode._agent_to_last_info["agent2"][
            "ncorrect2"
        ]
