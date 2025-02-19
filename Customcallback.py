from ray.rllib.algorithms.callbacks import DefaultCallbacks
from collections import defaultdict
import pdb


class CustomCallbacks(DefaultCallbacks):
    #     def on_episode_start(
    #         self, *, worker, base_env, policies, episode, env_index, **kwargs
    #     ):
    # #        episode.media["episode_data"] = defaultdict(list)
    #         #episode.user_data = {'agent1':{}, 'agent2':{}}

    #     def on_episode_step(
    #         self, *, worker, base_env, episode, env_index, **kwargs
    #     ):
    #         # Running metrics -> keep all values
    #         # Final metrics -> only keep the current value
    #         # for agent,_ in episode.user_data.items():
    #         #     pdb.set_trace()
    #         #     data = episode.last_info_for(agent)
    #         #     for name, value in data.items():
    #         #         episode.user_data[agent].append(value)
    #         #         else:
    #         #             data_subset[name] = value

    #         # # Arbitrary episode media
    #         # media = episode.last_info_for().get("media", {})
    #         # for name, value in media.items():
    #         #     episode.media["episode_data"][name].append(value)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        # for name, value in episode.media["episode_data"].items():
        #     episode.media["episode_data"][name] = np.array(value).tolist()
        # pdb.set_trace()
        # episode.custom_metrics = episode._agent_to_last_info
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
