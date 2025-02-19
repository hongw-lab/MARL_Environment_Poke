import gym
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, MultiDiscrete, Box, MultiBinary, Dict


### coop task ###
class MultiAgentSync_fullobs(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.randomize = config.get("randomize", True)
        self.randomize_miss = config.get("randomize_miss", False)
        self.width = config.get("width", 8)
        self.height = config.get("height", 8)
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        self.agent1_pos = config.get("Agent1", [0, 0])
        self.agent2_pos = config.get("Agent2", [0, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.movement_reward = config.get("movement_reward", -0.1)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.observation_space = Dict(
            {
                "nosepoke1": Discrete(self.width / 2),
                "water1": Discrete(self.width / 2),
                "otheragent1": Discrete(self.width / 2),
                "self1": Discrete(self.width / 2),
                "otherpoke1": Discrete(self.width / 2),
                "nosepoke0": Discrete(self.width / 2),
                "water0": Discrete(self.height),
                "otheragent0": Discrete(self.height),
                "self0": Discrete(self.height),
                "otherpoke0": Discrete(self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.

        if self.randomize:
            self.agent1_pos = [
                random.randint(0, self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.agent2_pos = [
                random.randint(0, self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.water_coords1 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.poke_coords1 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.water_coords2 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.poke_coords2 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(int(self.width / 2), self.width - 1),
            ]
        # punishment for movements

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # reward availability now
        self.water_available1 = 0
        self.water_available2 = 0
        self.gotwater1 = False
        self.gotwater2 = False
        self.poke1_recorded = False  # only record the first time agent pokes if the agent stays at the nose poke
        self.poke2_recorded = False
        self.sync_poke = 0
        self.sync_poke_last_step = 0
        self.nsync = 0
        self.miss = 0
        self.miss_last_step = 0
        self.poke_history1 = np.repeat(False, 5)  # poke in the last time point
        self.poke_history2 = np.repeat(False, 5)

        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False

        # info dict
        self.ncorrect = 0
        self.ncorrect1 = 0
        self.ncorrect2 = 0
        self.nmiss1 = 0
        self.nmiss2 = 0
        self.npoke1 = 0
        self.npoke2 = 0
        self.ndrink1 = 0
        self.ndrink2 = 0
        # How many timesteps have we done in this episode.
        self.timesteps = 0
        # generate roll out videos
        self.agent1_icon = "assets/agent1.png"
        self.agent2_icon = "assets/agent2.png"
        self.np_icon = "assets/nosepoke.png"
        self.water_icon = "assets/water.png"
        self.image_list = []
        self.grid_size = (self.width, self.height)
        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke_last_step = self.sync_poke
        self.sync_poke = 0
        self.miss_last_step = self.miss
        self.miss = 0
        self.gotwater1 = False
        self.gotwater2 = False
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1  # no action
            r1 = 0
            events1 = []
        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            if self.timeout:
                events1 = []
            if "drink" in events1:
                self.ndrink1 += 1
            elif "poke" in events1:
                self.npoke1 += 1
            if self.water_available1 == 1 and "drink" in events1:
                r1 = 2.0
                self.gotwater1 = True
                # self.drink1_list.append(self.timesteps)
                self.freeze1 = np.random.randint(4)
                self.water_available1 = 0
                if self.randomize:
                    self.water_coords1 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
                    self.poke_coords1 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
            else:
                r1 = self.movement_reward
        # if "poke" in events1:
        #     self.poke1_list.append(self.timesteps)

        if self.freeze2 > 0:
            self.freeze2 = self.freeze2 - 1.0  # no action
            r2 = 0
            events2 = []
        else:
            events2 = self._move(self.agent2_pos, action["agent2"], is_agent1=0)

            if self.timeout:
                events2 = []
            if "drink" in events2:
                self.ndrink2 += 1
            elif "poke" in events2:
                self.npoke2 += 1
            if self.water_available2 == 1.0 and "drink" in events2:
                r2 = 2.0
                self.gotwater2 = True
                # self.drink2_list.append(self.timesteps)
                self.freeze2 = np.random.randint(4)
                self.water_available2 = 0
                if self.randomize:
                    self.water_coords2 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
                    self.poke_coords2 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
            else:
                r2 = self.movement_reward

        # if "poke" in events2:
        #     self.poke2_list.append(self.timesteps)

        self.timeout = False
        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1

            # self.correct_list.append(self.timesteps)
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both
        elif np.any(poke_mat[:, 0]):
            r1 = -0.5
            r2 = -0.5
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            # self.timeout = True
            self.miss = 1
            if poke_mat[0, 0]:
                self.nmiss1 += 1
                # self.miss1_list.append(self.timesteps)
            elif poke_mat[1, 0]:
                self.nmiss2 += 1
                # self.miss2_list.append(self.timesteps)
            if self.randomize_miss:
                self.water_coords1 = [
                    random.randint(int(self.height / 2), self.height - 1),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
                self.poke_coords1 = [
                    random.randint(0, int(self.height / 2) - 2),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
                self.water_coords2 = [
                    random.randint(int(self.height / 2), self.height - 1),
                    random.randint(int(self.width / 2), self.width - 1),
                ]
                self.poke_coords2 = [
                    random.randint(0, int(self.height / 2) - 2),
                    random.randint(int(self.width / 2), self.width - 1),
                ]
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        if (
            self.sync_poke == 1
            and self.water_available1 == 0
            and self.water_available2 == 0
        ):  # cannot get double reward for sync poke
            r1 = 2.0
            r2 = 2.0
            self.ncorrect += 1
        if self.sync_poke == 1:
            self.water_available1 = 1
            self.water_available2 = 1
            self.nsync += 1

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }
        info = {
            "agent1": {
                "ncorrect": self.ncorrect,
                "ncorrect1": self.ncorrect1,
                "nmiss": self.nmiss1,
                "npoke1": self.npoke1,
                "ndrink1": self.ndrink1,
            },
            "agent2": {
                "ncorrect": self.ncorrect,
                "ncorrect2": self.ncorrect2,
                "nmiss": self.nmiss2,
                "npoke2": self.npoke2,
                "ndrink2": self.ndrink2,
            },
        }

        return obs, rewards, dones, info  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        return {
            "agent1": {
                "nosepoke0": self.poke_coords1[0],
                "water0": self.water_coords1[0],
                "self0": self.agent1_pos[0],
                "otheragent0": self.agent2_pos[0],
                "otherpoke0": self.poke_coords2[0],
                "nosepoke1": self.poke_coords1[1],
                "water1": self.water_coords1[1],
                "self1": self.agent1_pos[1],
                "otheragent1": self.agent2_pos[1] - 4,
                "otherpoke1": self.poke_coords2[1] - 4,
            },
            "agent2": {
                "nosepoke0": self.poke_coords2[0],
                "water0": self.water_coords2[0],
                "self0": self.agent2_pos[0],
                "otheragent0": self.agent1_pos[0],
                "otherpoke0": self.poke_coords1[0],
                "nosepoke1": self.poke_coords2[1] - 4,
                "water1": self.water_coords2[1] - 4,
                "self1": self.agent2_pos[1] - 4,
                "otheragent1": self.agent1_pos[1],
                "otherpoke1": self.poke_coords1[1],
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # check walls.
        # if is_agent1:
        #     bound = self.water_coords1
        # else:
        #     bound = self.water_coords2

        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        # elif coords[0] >= bound[0]:
        #     coords[0] = bound[0]
        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1
        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # nose poke
        if is_agent1:
            if coords == self.poke_coords1:
                if not self.poke1_recorded:
                    self.poke1_recorded = True
                    return {"poke"}
            else:
                self.poke1_recorded = False

            if coords == self.water_coords1:
                return {"drink"}
        else:
            if coords == self.poke_coords2:
                if not self.poke2_recorded:
                    self.poke2_recorded = True
                    return {"poke"}
            else:
                self.poke2_recorded = False

            if coords == self.water_coords2:
                return {"drink"}

        # no action
        if action == 4:
            return {"no_action"}

        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

    def render_image(self):
        # Set up the image size and grid parameters
        image_size = (414, 414)
        increment = 25
        grid_size = (8, 8)
        agent1_area_width = 4

        # Create a blank white image
        image = Image.new("RGB", (image_size[0] + 50, image_size[1] + 50), "white")
        draw = ImageDraw.Draw(image)

        # Calculate cell size
        cell_size = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

        # Draw grid
        for i in range(grid_size[0] + 1):
            draw.line(
                [
                    (i * cell_size[0] + increment, 0),
                    (i * cell_size[0] + increment, image_size[1]),
                ],
                fill="black",
            )

        for j in range(grid_size[1] + 1):
            draw.line(
                [
                    (0 + increment, j * cell_size[1]),
                    (image_size[0] + increment, j * cell_size[1]),
                ],
                fill="black",
            )

        # Highlight agent areas
        agent1_area = (
            0 + increment,
            0,
            agent1_area_width * cell_size[0] + increment,
            image_size[1],
        )
        agent2_area = (
            agent1_area_width * cell_size[0] + increment,
            0,
            image_size[0] + increment,
            image_size[1],
        )

        draw.rectangle(agent1_area, outline="black", width=2)
        draw.rectangle(agent2_area, outline="black", width=2)

        # Add text information
        fontpath = "/home/kumquat/Downloads/aileron_0102/Aileron-Bold.otf"
        new_font_size = 20  # Adjust the size as needed
        font = ImageFont.truetype(fontpath, new_font_size)

        text_y_position = image_size[1] + 25

        text = (
            f"!Correct Trial!"
            if self.sync_poke == 1 or self.sync_poke_last_step == 1
            else "!Miss!" if self.miss == 1 or self.miss_last_step == 1 else ""
        )
        draw.text((170, text_y_position - 10), text, font=font, fill="black")
        # Move text information to the bottom
        self.draw_event_text(
            draw, self.events["agent1"], (30, text_y_position), font, "red"
        )
        self.draw_event_text(
            draw, self.events["agent2"], (290, text_y_position), font, "blue"
        )

        # Draw water and nose poke coordinates
        self.draw_symbol(image, self.water_icon, self.water_coords1, cell_size, increment)
        self.draw_symbol(image, self.water_icon, self.water_coords2, cell_size, increment)
        self.draw_symbol(image, self.np_icon, self.poke_coords1, cell_size, increment)
        self.draw_symbol(image, self.np_icon, self.poke_coords2, cell_size, increment)
        # Draw agent positions
        self.draw_symbol(image, self.agent1_icon, self.agent1_pos, cell_size, increment)
        self.draw_symbol(image, self.agent2_icon, self.agent2_pos, cell_size, increment)
        # Display the updated image
        self.image_list.append(image)

    def draw_symbol(self, image, symbol, position, cell_size, increment):
        # Calculate the center of the cell
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]

        icon = Image.open(symbol).convert("RGBA")
        icon_size = (
            int(cell_size[0] * 0.8),
            int(cell_size[1] * 0.8),
        )
        icon.thumbnail(icon_size, Image.LANCZOS)
        top_left_x = int(center_x - icon.width / 2)
        top_left_y = int(center_y - icon.height / 2)
        image.paste(icon, (top_left_x, top_left_y), icon)


    def draw_event_text(self, draw, events, position, font, color):
        text = (
            "!Agent Poke!"
            if "poke" in events
            else "!Agent Drink!" if "drink" in events else ""
        )
        draw.text(position, text, font=font, fill=color)

        ##############################################


### single task ###
class MultiAgentSing_fullobs(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.randomize = config.get("randomize", True)
        self.width = config.get("width", 8)
        self.height = config.get("height", 6)
        self.vision = config.get("Vision", [1, 2])
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        self.agent1_pos = config.get("Agent1", [0, 0])
        self.agent2_pos = config.get("Agent2", [0, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.movement_reward = config.get("movement_reward", -0.1)
        self.observation_space = Dict(
            {
                "nosepoke1": Discrete(self.width / 2),
                "water1": Discrete(self.width / 2),
                "otheragent1": Discrete(self.width / 2),
                "self1": Discrete(self.width / 2),
                "otherpoke1": Discrete(self.width / 2),
                "nosepoke0": Discrete(self.width / 2),
                "water0": Discrete(self.height),
                "otheragent0": Discrete(self.height),
                "self0": Discrete(self.height),
                "otherpoke0": Discrete(self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        if self.randomize:
            self.agent1_pos = [
                random.randint(0, self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.agent2_pos = [
                random.randint(0, self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.water_coords1 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.poke_coords1 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.water_coords2 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.poke_coords2 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(int(self.width / 2), self.width - 1),
            ]
        # reward availability now
        self.agent1_R = 0.0
        self.agent2_R = 0.0
        self.water_available1 = False
        self.water_available2 = False
        self.gotwater1 = False
        self.gotwater2 = False

        self.ncorrect = 0
        self.ncorrect1 = 0
        self.ncorrect2 = 0
        self.nmiss1 = 0
        self.nmiss2 = 0
        self.npoke1 = 0
        self.npoke2 = 0
        self.ndrink1 = 0
        self.ndrink2 = 0

        self.sync_poke = 0
        self.miss = 0
        self.poke_history1 = np.array([False, False])  # poke in the last time point
        self.poke_history2 = np.array([False, False])
        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False

        self.poke1_recorded = False  # only record the first time agent pokes if the agent stays at the nose poke
        self.poke2_recorded = False

        self.agent1_icon = "assets/agent1.png"
        self.agent2_icon = "assets/agent2.png"
        self.np_icon = "assets/nosepoke.png"
        self.water_icon = "assets/water.png"
        self.timesteps = 0
        self.image_list = []
        self.grid_size = (self.width, self.height)
        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke = 0
        self.miss = 0
        self.gotwater1 = False
        self.gotwater2 = False
        # print(self.water_available1)
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit
        # print(self.poke2_recorded)
        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1.0  # no action
            r1 = 0
            events1 = []

        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            if self.timeout:
                events1 = []
            if "drink" in events1:
                self.ndrink1 += 1
            elif "poke" in events1:
                self.npoke1 += 1
            if self.water_available1 and "drink" in events1:
                r1 = 2.0
                self.ncorrect1 += 1
                self.gotwater1 = True
                # self.drink1_list.append(self.timesteps)
                self.freeze1 = np.random.randint(4)
                self.water_available1 = False
                if self.randomize:
                    self.water_coords1 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
                    self.poke_coords1 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
            elif (not self.water_available1) and "poke" in events1:
                r1 = 2.0
                self.water_available1 = True
                # self.poke1_list.append(self.timesteps)

            # elif "no_action" in events1:
            #     r1 = -0.1
            else:
                r1 = self.movement_reward

        if self.freeze2 > 0:
            self.freeze2 = self.freeze2 - 1.0  # no action
            r2 = 0
            events2 = []
        else:
            events2 = self._move(self.agent2_pos, action["agent2"], is_agent1=0)

            if self.timeout:
                events2 = []
            if "drink" in events2:
                self.ndrink2 += 1
            elif "poke" in events2:
                self.npoke2 += 1
            if self.water_available2 and "drink" in events2:
                r2 = 2.0
                self.ncorrect2 += 1
                self.gotwater2 = True
                # self.drink2_list.append(self.timesteps)
                self.freeze2 = np.random.randint(4)
                self.water_available2 = False
                if self.randomize:
                    self.water_coords2 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]  # upper left corner
                    self.poke_coords2 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]  # upper left corner
            elif (not self.water_available2) and "poke" in events2:
                r2 = 2.0
                self.water_available2 = True
                # self.poke2_list.append(self.timesteps)
            # elif "no_action" in events2:
            #     r2 = -0.1
            else:
                r2 = self.movement_reward

        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1
            self.ncorrect += 1
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both and create time out - any poking in the next time step does not count
        elif np.any(poke_mat[:, 0]):
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            self.miss = 1
            if poke_mat[0, 0]:
                self.nmiss1 += 1
            elif poke_mat[1, 0]:
                self.nmiss2 += 1
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }

        info = {
            "agent1": {
                "ncorrect": self.ncorrect,
                "ncorrect1": self.ncorrect1,
                "nmiss": self.nmiss1,
                "npoke1": self.npoke1,
                "ndrink1": self.ndrink1,
            },
            "agent2": {
                "ncorrect": self.ncorrect,
                "ncorrect2": self.ncorrect2,
                "nmiss": self.nmiss2,
                "npoke2": self.npoke2,
                "ndrink2": self.ndrink2,
            },
        }
        return obs, rewards, dones, info  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        # discrete coordinate of the locations of nose poke water port and
        return {
            "agent1": {
                "nosepoke0": self.poke_coords1[0],
                "water0": self.water_coords1[0],
                "self0": self.agent1_pos[0],
                "otheragent0": self.agent2_pos[0],
                "otherpoke0": self.poke_coords2[0],
                "nosepoke1": self.poke_coords1[1],
                "water1": self.water_coords1[1],
                "self1": self.agent1_pos[1],
                "otheragent1": self.agent2_pos[1] - 4,
                "otherpoke1": self.poke_coords2[1] - 4,
            },
            "agent2": {
                "nosepoke0": self.poke_coords2[0],
                "water0": self.water_coords2[0],
                "self0": self.agent2_pos[0],
                "otheragent0": self.agent1_pos[0],
                "otherpoke0": self.poke_coords1[0],
                "nosepoke1": self.poke_coords2[1] - 4,
                "water1": self.water_coords2[1] - 4,
                "self1": self.agent2_pos[1] - 4,
                "otheragent1": self.agent1_pos[1],
                "otherpoke1": self.poke_coords1[1],
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # check walls.

        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1

        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1

        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # update agent location
        # nose poke
        if is_agent1:
            if coords == self.poke_coords1:
                if not self.poke1_recorded:
                    self.poke1_recorded = True
                    return {"poke"}
            else:
                self.poke1_recorded = False

            if coords == self.water_coords1:
                return {"drink"}
        else:
            if coords == self.poke_coords2:
                if not self.poke2_recorded:
                    self.poke2_recorded = True
                    return {"poke"}
            else:
                self.poke2_recorded = False

            if coords == self.water_coords2:
                return {"drink"}

        # no action
        if action == 4:
            return {"no_action"}
        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

    def render_image(self):
        # Set up the image size and grid parameters
        image_size = (414, 414)
        increment = 25
        grid_size = (8, 8)
        agent1_area_width = 4

        # Create a blank white image
        image = Image.new("RGB", (image_size[0] + 50, image_size[1] + 50), "white")
        draw = ImageDraw.Draw(image)

        # Calculate cell size
        cell_size = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

        # Draw grid
        for i in range(grid_size[0] + 1):
            draw.line(
                [
                    (i * cell_size[0] + increment, 0),
                    (i * cell_size[0] + increment, image_size[1]),
                ],
                fill="black",
            )

        for j in range(grid_size[1] + 1):
            draw.line(
                [
                    (0 + increment, j * cell_size[1]),
                    (image_size[0] + increment, j * cell_size[1]),
                ],
                fill="black",
            )

        # Highlight agent areas
        agent1_area = (
            0 + increment,
            0,
            agent1_area_width * cell_size[0] + increment,
            image_size[1],
        )
        agent2_area = (
            agent1_area_width * cell_size[0] + increment,
            0,
            image_size[0] + increment,
            image_size[1],
        )

        draw.rectangle(agent1_area, outline="black", width=2)
        draw.rectangle(agent2_area, outline="black", width=2)

        # Add text information
        fontpath = "/home/kumquat/Downloads/aileron_0102/Aileron-Bold.otf"
        new_font_size = 20  # Adjust the size as needed
        font = ImageFont.truetype(fontpath, new_font_size)

        text_y_position = image_size[1] + 25

        text = (
            f"!SyncPoke!" if self.sync_poke == 1 else "!UnsyncPoke!" if self.miss == 1 else ""
        )
        draw.text((170, text_y_position - 10), text, font=font, fill="black")
        # Move text information to the bottom
        self.draw_event_text(
            draw, self.events["agent1"], (30, text_y_position), font, "red"
        )
        self.draw_event_text(
            draw, self.events["agent2"], (290, text_y_position), font, "blue"
        )

         # Draw water and nose poke coordinates
        self.draw_symbol(image, self.water_icon, self.water_coords1, cell_size, increment)
        self.draw_symbol(image, self.water_icon, self.water_coords2, cell_size, increment)
        self.draw_symbol(image, self.np_icon, self.poke_coords1, cell_size, increment)
        self.draw_symbol(image, self.np_icon, self.poke_coords2, cell_size, increment)
        # Draw agent positions
        self.draw_symbol(image, self.agent1_icon, self.agent1_pos, cell_size, increment)
        self.draw_symbol(image, self.agent2_icon, self.agent2_pos, cell_size, increment)
        # Display the updated image
        self.image_list.append(image)

    def draw_symbol(self, image, symbol, position, cell_size, increment):
        # Calculate the center of the cell
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]

        icon = Image.open(symbol).convert("RGBA")
        icon_size = (
            int(cell_size[0] * 0.8),
            int(cell_size[1] * 0.8),
        )
        icon.thumbnail(icon_size, Image.LANCZOS)
        top_left_x = int(center_x - icon.width / 2)
        top_left_y = int(center_y - icon.height / 2)
        image.paste(icon, (top_left_x, top_left_y), icon)

    def draw_event_text(self, draw, events, position, font, color):
        text = (
            "!Agent Poke!"
            if "poke" in events
            else "!Agent Drink!" if "drink" in events else ""
        )
        draw.text(position, text, font=font, fill=color)

        ######################################


### only one side is learning, the other side is trained for single ###
class MultiAgentSync_oneside(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.randomize = config.get("randomize", True)
        self.width = config.get("width", 8)
        self.height = config.get("height", 6)
        self.vision = config.get("Vision", [1, 2])
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        self.agent1_pos = config.get("Agent1", [0, 0])
        self.agent2_pos = config.get("Agent2", [0, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.movement_reward = config.get("movement_reward", -0.1)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.observation_space = Dict(
            {
                "nosepoke": Discrete(self.width * self.height / 2),
                "water": Discrete(self.width * self.height / 2),
                "otheragent": Discrete(self.width * self.height / 2),
                # 0 is unknown location, if known, block 1 to the last
                "self": Discrete(self.width * self.height / 2),
                "otherpoke": Discrete(self.width * self.height / 2),
                # "noise" : Discrete(self.width*self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.

        if self.randomize:
            self.agent1_pos = [
                random.randint(0, self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.agent2_pos = [
                random.randint(0, self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.water_coords1 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.poke_coords1 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.water_coords2 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.poke_coords2 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(int(self.width / 2), self.width - 1),
            ]
        # punishment for movements

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # reward availability now
        self.water_available1 = 0
        self.water_available2 = 0
        self.gotwater1 = False
        self.gotwater2 = False
        self.poke1_recorded = False  # only record the first time agent pokes if the agent stays at the nose poke
        self.poke2_recorded = False
        self.sync_poke = 0
        self.miss = 0
        self.poke_history1 = np.repeat(False, 5)  # poke in the last time point
        self.poke_history2 = np.repeat(False, 5)

        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False

        # info dict
        self.ncorrect = 0
        self.ncorrect1 = 0
        self.ncorrect2 = 0
        self.nmiss1 = 0
        self.nmiss2 = 0
        self.npoke1 = 0
        self.npoke2 = 0
        self.ndrink1 = 0
        self.ndrink2 = 0
        # How many timesteps have we done in this episode.
        self.timesteps = 0
        self.image_list = []
        self.grid_size = (self.width, self.height)
        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke = 0
        self.miss = 0
        self.gotwater1 = False
        self.gotwater2 = False
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1.0  # no action
            r1 = 0
            events1 = []
        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            if self.timeout:
                events1 = []
            if "drink" in events1:
                self.ndrink1 += 1
            elif "poke" in events1:
                self.npoke1 += 1
            if self.water_available1 == 1.0 and "drink" in events1:
                r1 = 2.0
                self.gotwater1 = True
                # self.drink1_list.append(self.timesteps)
                self.freeze1 = np.random.randint(4)
                self.water_available1 = 0
                if self.randomize:
                    self.water_coords1 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
                    self.poke_coords1 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
            elif (not self.water_available1) and "poke" in events1:
                r1 = 2.0
                self.water_available1 = True
            else:
                r1 = self.movement_reward
        # if "poke" in events1:
        #     self.poke1_list.append(self.timesteps)

        if self.freeze2 > 0:
            self.freeze2 = self.freeze2 - 1.0  # no action
            r2 = 0
            events2 = []
        else:
            events2 = self._move(self.agent2_pos, action["agent2"], is_agent1=0)

            if self.timeout:
                events2 = []
            if "drink" in events2:
                self.ndrink2 += 1
            elif "poke" in events2:
                self.npoke2 += 1
            if self.water_available2 == 1.0 and "drink" in events2:
                r2 = 2.0
                self.gotwater2 = True
                # self.drink2_list.append(self.timesteps)
                self.freeze2 = np.random.randint(4)
                self.water_available2 = 0
                if self.randomize:
                    self.water_coords2 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
                    self.poke_coords2 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
            else:
                r2 = self.movement_reward

        # if "poke" in events2:
        #     self.poke2_list.append(self.timesteps)

        self.timeout = False
        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1
            self.ncorrect += 1
            # self.correct_list.append(self.timesteps)
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both and create time out - any poking in the next time step does not count
        elif np.any(poke_mat[:, 0]):
            r1 = -0.5
            r2 = -0.5
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            # self.timeout = True
            self.miss = 1
            if poke_mat[0, 0]:
                self.nmiss1 += 1
                # self.miss1_list.append(self.timesteps)
            elif poke_mat[1, 0]:
                self.nmiss2 += 1
                # self.miss2_list.append(self.timesteps)
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        if (
            self.sync_poke == 1 and self.water_available2 == 0
        ):  # cannot get double reward for sync poke
            r2 = 2.0
        if self.sync_poke == 1:
            self.water_available2 = 1.0

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }
        info = {
            "agent1": {
                "ncorrect": self.ncorrect,
                "ncorrect1": self.ncorrect1,
                "nmiss": self.nmiss1,
                "npoke1": self.npoke1,
                "ndrink1": self.ndrink1,
            },
            "agent2": {
                "ncorrect": self.ncorrect,
                "ncorrect2": self.ncorrect2,
                "nmiss": self.nmiss2,
                "npoke2": self.npoke2,
                "ndrink2": self.ndrink2,
            },
        }

        return obs, rewards, dones, info  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        # discrete coordinate of the locations of nose poke water port and
        nosepoke_pos1 = (
            int(
                self.poke_coords1[0] * self.width / 2
                + (self.poke_coords1[1] % (self.width / 2) + 1)
            )
            - 1
        )
        nosepoke_pos2 = (
            int(
                self.poke_coords2[0] * self.width / 2
                + (self.poke_coords2[1] % (self.width / 2) + 1)
            )
            - 1
        )
        water_pos1 = (
            int(
                self.water_coords1[0] * self.width / 2
                + (self.water_coords1[1] % (self.width / 2) + 1)
            )
            - 1
        )
        water_pos2 = (
            int(
                self.water_coords2[0] * self.width / 2
                + (self.water_coords2[1] % (self.width / 2) + 1)
            )
            - 1
        )
        ag1_discrete_pos = (
            int(
                self.agent1_pos[0] * self.width / 2
                + (self.agent1_pos[1] % (self.width / 2) + 1)
            )
            - 1
        )
        ag2_discrete_pos = (
            int(
                self.agent2_pos[0] * self.width / 2
                + (self.agent2_pos[1] % (self.width / 2) + 1)
            )
            - 1
        )

        ag2other = ag1_discrete_pos
        ag1other = ag2_discrete_pos

        # noise1 = random.randint(1, self.width*self.height)
        # noise2 = random.randint(1, self.width*self.height)

        return {
            "agent1": {
                "nosepoke": nosepoke_pos1,
                "water": water_pos1,
                "self": ag1_discrete_pos,
                "otheragent": ag1other,
                "otherpoke": nosepoke_pos2,
            },
            "agent2": {
                "nosepoke": nosepoke_pos2,
                "water": water_pos2,
                "self": ag2_discrete_pos,
                "otheragent": ag2other,
                "otherpoke": nosepoke_pos1,
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # check walls.
        if is_agent1:
            bound = self.water_coords1
        else:
            bound = self.water_coords2

        if coords[0] < 0:
            coords[0] = 0
        # elif coords[0] >= self.height:
        #     coords[0] = self.height - 1
        elif coords[0] >= bound[0]:
            coords[0] = bound[0]
        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1
        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # nose poke
        if is_agent1:
            if coords == self.poke_coords1:
                if not self.poke1_recorded:
                    self.poke1_recorded = True
                    return {"poke"}
            else:
                self.poke1_recorded = False

            if coords == self.water_coords1:
                return {"drink"}
        else:
            if coords == self.poke_coords2:
                if not self.poke2_recorded:
                    self.poke2_recorded = True
                    return {"poke"}
            else:
                self.poke2_recorded = False

            if coords == self.water_coords2:
                return {"drink"}

        # no action
        if action == 4:
            return {"no_action"}

        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

    def render_image(self):
        # Set up the image size and grid parameters
        image_size = (400, 400)
        increment = 25
        grid_size = (8, 8)
        agent1_area_width = 4

        # Create a blank white image
        image = Image.new("RGB", (image_size[0] + 50, image_size[1] + 50), "white")
        draw = ImageDraw.Draw(image)

        # Calculate cell size
        cell_size = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

        # Draw grid
        for i in range(grid_size[0] + 1):
            draw.line(
                [
                    (i * cell_size[0] + increment, 0),
                    (i * cell_size[0] + increment, image_size[1]),
                ],
                fill="black",
            )

        for j in range(grid_size[1] + 1):
            draw.line(
                [
                    (0 + increment, j * cell_size[1]),
                    (image_size[0] + increment, j * cell_size[1]),
                ],
                fill="black",
            )

        # Highlight agent areas
        agent1_area = (
            0 + increment,
            0,
            agent1_area_width * cell_size[0] + increment,
            image_size[1],
        )
        agent2_area = (
            agent1_area_width * cell_size[0] + increment,
            0,
            image_size[0] + increment,
            image_size[1],
        )

        draw.rectangle(agent1_area, outline="black", width=2)
        draw.rectangle(agent2_area, outline="black", width=2)

        # Add text information
        fontpath = "/home/kumquat/Downloads/aileron_0102/Aileron-Bold.otf"
        new_font_size = 20  # Adjust the size as needed
        font = ImageFont.truetype(fontpath, new_font_size)

        text_y_position = image_size[1] + 25

        text = (
            f"!SyncPoke!" if self.sync_poke == 1 else "!Miss!" if self.miss == 1 else ""
        )
        draw.text((170, text_y_position - 10), text, font=font, fill="black")
        # Move text information to the bottom
        self.draw_event_text(
            draw, self.events["agent1"], (30, text_y_position), font, "red"
        )
        self.draw_event_text(
            draw, self.events["agent2"], (290, text_y_position), font, "blue"
        )

        # Draw "o" at nose poke coordinates
        self.draw_symbol(
            draw, "N", font, self.poke_coords1, cell_size, increment, "yellow"
        )
        self.draw_symbol(
            draw, "N", font, self.poke_coords2, cell_size, increment, "yellow"
        )

        # Draw star at water coordinates
        self.draw_symbol(
            draw, "W", font, self.water_coords1, cell_size, increment, "green"
        )
        self.draw_symbol(
            draw, "W", font, self.water_coords2, cell_size, increment, "green"
        )
        # Draw agent positions
        self.draw_agent_position(
            draw, self.agent1_pos, increment, cell_size, color="red"
        )
        self.draw_agent_position(
            draw, self.agent2_pos, increment, cell_size, color="blue"
        )
        # Display the updated image
        self.image_list.append(image)
        # If you want to save the image, you can use self.image.save("filename.png")

    def draw_symbol(self, draw, symbol, font, position, cell_size, increment, color):
        # Calculate the center of the cell
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]
        left = position[1] * cell_size[0] + increment
        top = position[0] * cell_size[1]
        right = left + cell_size[0]
        bottom = top + cell_size[1]
        # paint the tile
        draw.rectangle([left, top, right, bottom], fill=color)
        # Draw the symbol at the center
        draw.text((center_x, center_y), symbol, font=font, fill="black", anchor="mm")

    def draw_agent_position(self, draw, position, increment, cell_size, color):
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]
        radius = (
            min(cell_size) / 4
        )  # Calculate radius as half of the minimum cell dimension

        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=color,
        )

    def draw_event_text(self, draw, events, position, font, color):
        text = (
            "!Agent Poke!"
            if "poke" in events
            else "!Agent Drink!" if "drink" in events else ""
        )
        draw.text(position, text, font=font, fill=color)


### control - multi agent with no obs ###
class MultiAgentSync_noobs(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.randomize = config.get("randomize", True)
        self.randomize_miss = config.get("randomize_miss", False)
        self.width = config.get("width", 8)
        self.height = config.get("height", 8)
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        self.agent1_pos = config.get("Agent1", [0, 0])
        self.agent2_pos = config.get("Agent2", [0, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.movement_reward = config.get("movement_reward", -0.1)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.observation_space = Dict(
            {
                "nosepoke1": Discrete(self.width / 2),
                "water1": Discrete(self.width / 2),
                # "otheragent1": Discrete(self.width / 2),
                "self1": Discrete(self.width / 2),
                # "otherpoke1": Discrete(self.width / 2),
                "nosepoke0": Discrete(self.width / 2),
                "water0": Discrete(self.height),
                # "otheragent0": Discrete(self.height),
                "self0": Discrete(self.height),
                # "otherpoke0": Discrete(self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.

        if self.randomize:
            self.agent1_pos = [
                random.randint(0, self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.agent2_pos = [
                random.randint(0, self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.water_coords1 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.poke_coords1 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.water_coords2 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.poke_coords2 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(int(self.width / 2), self.width - 1),
            ]
        # punishment for movements

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # reward availability now
        self.water_available1 = 0
        self.water_available2 = 0
        self.gotwater1 = False
        self.gotwater2 = False
        self.poke1_recorded = False  # only record the first time agent pokes if the agent stays at the nose poke
        self.poke2_recorded = False
        self.sync_poke = 0
        self.nsync = 0
        self.miss = 0
        self.poke_history1 = np.repeat(False, 5)  # poke in the last time point
        self.poke_history2 = np.repeat(False, 5)

        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False

        # info dict
        self.ncorrect = 0
        self.ncorrect1 = 0
        self.ncorrect2 = 0
        self.nmiss1 = 0
        self.nmiss2 = 0
        self.npoke1 = 0
        self.npoke2 = 0
        self.ndrink1 = 0
        self.ndrink2 = 0
        # How many timesteps have we done in this episode.
        self.timesteps = 0

        self.image_list = []
        self.grid_size = (self.width, self.height)
        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke = 0
        self.miss = 0
        self.gotwater1 = False
        self.gotwater2 = False
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1  # no action
            r1 = 0
            events1 = []
        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            if self.timeout:
                events1 = []
            if "drink" in events1:
                self.ndrink1 += 1
            elif "poke" in events1:
                self.npoke1 += 1
            if self.water_available1 == 1 and "drink" in events1:
                r1 = 2.0
                self.gotwater1 = True
                # self.drink1_list.append(self.timesteps)
                self.freeze1 = np.random.randint(4)
                self.water_available1 = 0
                if self.randomize:
                    self.water_coords1 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
                    self.poke_coords1 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
            else:
                r1 = self.movement_reward
        # if "poke" in events1:
        #     self.poke1_list.append(self.timesteps)

        if self.freeze2 > 0:
            self.freeze2 = self.freeze2 - 1.0  # no action
            r2 = 0
            events2 = []
        else:
            events2 = self._move(self.agent2_pos, action["agent2"], is_agent1=0)

            if self.timeout:
                events2 = []
            if "drink" in events2:
                self.ndrink2 += 1
            elif "poke" in events2:
                self.npoke2 += 1
            if self.water_available2 == 1.0 and "drink" in events2:
                r2 = 2.0
                self.gotwater2 = True
                # self.drink2_list.append(self.timesteps)
                self.freeze2 = np.random.randint(4)
                self.water_available2 = 0
                if self.randomize:
                    self.water_coords2 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
                    self.poke_coords2 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
            else:
                r2 = self.movement_reward

        # if "poke" in events2:
        #     self.poke2_list.append(self.timesteps)

        self.timeout = False
        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1

            # self.correct_list.append(self.timesteps)
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both
        elif np.any(poke_mat[:, 0]):
            r1 = -0.5
            r2 = -0.5
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            # self.timeout = True
            self.miss = 1
            if poke_mat[0, 0]:
                self.nmiss1 += 1
                # self.miss1_list.append(self.timesteps)
            elif poke_mat[1, 0]:
                self.nmiss2 += 1
                # self.miss2_list.append(self.timesteps)
            if self.randomize_miss:
                self.water_coords1 = [
                    random.randint(int(self.height / 2), self.height - 1),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
                self.poke_coords1 = [
                    random.randint(0, int(self.height / 2) - 2),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
                self.water_coords2 = [
                    random.randint(int(self.height / 2), self.height - 1),
                    random.randint(int(self.width / 2), self.width - 1),
                ]
                self.poke_coords2 = [
                    random.randint(0, int(self.height / 2) - 2),
                    random.randint(int(self.width / 2), self.width - 1),
                ]
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        if (
            self.sync_poke == 1
            and self.water_available1 == 0
            and self.water_available2 == 0
        ):  # cannot get double reward for sync poke
            r1 = 2.0
            r2 = 2.0
            self.ncorrect += 1
        if self.sync_poke == 1:
            self.water_available1 = 1
            self.water_available2 = 1
            self.nsync += 1

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }
        info = {
            "agent1": {
                "ncorrect": self.ncorrect,
                "ncorrect1": self.ncorrect1,
                "nmiss": self.nmiss1,
                "npoke1": self.npoke1,
                "ndrink1": self.ndrink1,
            },
            "agent2": {
                "ncorrect": self.ncorrect,
                "ncorrect2": self.ncorrect2,
                "nmiss": self.nmiss2,
                "npoke2": self.npoke2,
                "ndrink2": self.ndrink2,
            },
        }

        return obs, rewards, dones, info  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        return {
            "agent1": {
                "nosepoke0": self.poke_coords1[0],
                "water0": self.water_coords1[0],
                "self0": self.agent1_pos[0],
                # "otheragent0": self.agent2_pos[0],
                # "otherpoke0": self.poke_coords2[0],
                "nosepoke1": self.poke_coords1[1],
                "water1": self.water_coords1[1],
                "self1": self.agent1_pos[1],
                # "otheragent1": self.agent2_pos[1] - 4,
                # otherpoke1": self.poke_coords2[1] - 4,
            },
            "agent2": {
                "nosepoke0": self.poke_coords2[0],
                "water0": self.water_coords2[0],
                "self0": self.agent2_pos[0],
                # "otheragent0": self.agent1_pos[0],
                # "otherpoke0": self.poke_coords1[0],
                "nosepoke1": self.poke_coords2[1] - 4,
                "water1": self.water_coords2[1] - 4,
                "self1": self.agent2_pos[1] - 4,
                # "otheragent1": self.agent1_pos[1],
                # "otherpoke1": self.poke_coords1[1],
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # check walls.
        # if is_agent1:
        #     bound = self.water_coords1
        # else:
        #     bound = self.water_coords2

        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        # elif coords[0] >= bound[0]:
        #     coords[0] = bound[0]
        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1
        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # nose poke
        if is_agent1:
            if coords == self.poke_coords1:
                if not self.poke1_recorded:
                    self.poke1_recorded = True
                    return {"poke"}
            else:
                self.poke1_recorded = False

            if coords == self.water_coords1:
                return {"drink"}
        else:
            if coords == self.poke_coords2:
                if not self.poke2_recorded:
                    self.poke2_recorded = True
                    return {"poke"}
            else:
                self.poke2_recorded = False

            if coords == self.water_coords2:
                return {"drink"}

        # no action
        if action == 4:
            return {"no_action"}

        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

    def render_image(self):
        # Set up the image size and grid parameters
        image_size = (400, 400)
        increment = 25
        grid_size = (8, 8)
        agent1_area_width = 4

        # Create a blank white image
        image = Image.new("RGB", (image_size[0] + 50, image_size[1] + 50), "white")
        draw = ImageDraw.Draw(image)

        # Calculate cell size
        cell_size = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

        # Draw grid
        for i in range(grid_size[0] + 1):
            draw.line(
                [
                    (i * cell_size[0] + increment, 0),
                    (i * cell_size[0] + increment, image_size[1]),
                ],
                fill="black",
            )

        for j in range(grid_size[1] + 1):
            draw.line(
                [
                    (0 + increment, j * cell_size[1]),
                    (image_size[0] + increment, j * cell_size[1]),
                ],
                fill="black",
            )

        # Highlight agent areas
        agent1_area = (
            0 + increment,
            0,
            agent1_area_width * cell_size[0] + increment,
            image_size[1],
        )
        agent2_area = (
            agent1_area_width * cell_size[0] + increment,
            0,
            image_size[0] + increment,
            image_size[1],
        )

        draw.rectangle(agent1_area, outline="black", width=2)
        draw.rectangle(agent2_area, outline="black", width=2)

        # Add text information
        fontpath = "/home/kumquat/Downloads/aileron_0102/Aileron-Bold.otf"
        new_font_size = 20  # Adjust the size as needed
        font = ImageFont.truetype(fontpath, new_font_size)

        text_y_position = image_size[1] + 25

        text = (
            f"!SyncPoke!" if self.sync_poke == 1 else "!Miss!" if self.miss == 1 else ""
        )
        draw.text((170, text_y_position - 10), text, font=font, fill="black")
        # Move text information to the bottom
        self.draw_event_text(
            draw, self.events["agent1"], (30, text_y_position), font, "red"
        )
        self.draw_event_text(
            draw, self.events["agent2"], (290, text_y_position), font, "blue"
        )

        # Draw "o" at nose poke coordinates
        self.draw_symbol(
            draw, "N", font, self.poke_coords1, cell_size, increment, "yellow"
        )
        self.draw_symbol(
            draw, "N", font, self.poke_coords2, cell_size, increment, "yellow"
        )

        # Draw star at water coordinates
        self.draw_symbol(
            draw, "W", font, self.water_coords1, cell_size, increment, "green"
        )
        self.draw_symbol(
            draw, "W", font, self.water_coords2, cell_size, increment, "green"
        )
        # Draw agent positions
        self.draw_agent_position(
            draw, self.agent1_pos, increment, cell_size, color="red"
        )
        self.draw_agent_position(
            draw, self.agent2_pos, increment, cell_size, color="blue"
        )
        # Display the updated image
        self.image_list.append(image)
        # If you want to save the image, you can use self.image.save("filename.png")

    def draw_symbol(self, draw, symbol, font, position, cell_size, increment, color):
        # Calculate the center of the cell
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]
        left = position[1] * cell_size[0] + increment
        top = position[0] * cell_size[1]
        right = left + cell_size[0]
        bottom = top + cell_size[1]
        # paint the tile
        draw.rectangle([left, top, right, bottom], fill=color)
        # Draw the symbol at the center
        draw.text((center_x, center_y), symbol, font=font, fill="black", anchor="mm")

    def draw_agent_position(self, draw, position, increment, cell_size, color):
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]
        radius = (
            min(cell_size) / 4
        )  # Calculate radius as half of the minimum cell dimension

        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=color,
        )

    def draw_event_text(self, draw, events, position, font, color):
        text = (
            "!Agent Poke!"
            if "poke" in events
            else "!Agent Drink!" if "drink" in events else ""
        )
        draw.text(position, text, font=font, fill=color)


### single task no obs control ###
class MultiAgentSing_noobs(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.randomize = config.get("randomize", True)
        self.width = config.get("width", 8)
        self.height = config.get("height", 6)
        self.vision = config.get("Vision", [1, 2])
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        self.agent1_pos = config.get("Agent1", [0, 0])
        self.agent2_pos = config.get("Agent2", [0, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.movement_reward = config.get("movement_reward", -0.1)
        self.observation_space = Dict(
            {
                "nosepoke1": Discrete(self.width / 2),
                "water1": Discrete(self.width / 2),
                # "otheragent1": Discrete(self.width / 2),
                "self1": Discrete(self.width / 2),
                # "otherpoke1": Discrete(self.width / 2),
                "nosepoke0": Discrete(self.width / 2),
                "water0": Discrete(self.height),
                # "otheragent0": Discrete(self.height),
                "self0": Discrete(self.height),
                # "otherpoke0": Discrete(self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        if self.randomize:
            self.agent1_pos = [
                random.randint(0, self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.agent2_pos = [
                random.randint(0, self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.water_coords1 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.poke_coords1 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.water_coords2 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.poke_coords2 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(int(self.width / 2), self.width - 1),
            ]
        # reward availability now
        self.agent1_R = 0.0
        self.agent2_R = 0.0
        self.water_available1 = False
        self.water_available2 = False
        self.gotwater1 = False
        self.gotwater2 = False

        self.ncorrect = 0
        self.ncorrect1 = 0
        self.ncorrect2 = 0
        self.nmiss1 = 0
        self.nmiss2 = 0
        self.npoke1 = 0
        self.npoke2 = 0
        self.ndrink1 = 0
        self.ndrink2 = 0

        self.sync_poke = 0
        self.miss = 0
        self.poke_history1 = np.array([False, False])  # poke in the last time point
        self.poke_history2 = np.array([False, False])
        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False

        self.poke1_recorded = False  # only record the first time agent pokes if the agent stays at the nose poke
        self.poke2_recorded = False

        self.timesteps = 0
        self.image_list = []
        self.grid_size = (self.width, self.height)
        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke = 0
        self.miss = 0
        self.gotwater1 = False
        self.gotwater2 = False
        # print(self.water_available1)
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit
        # print(self.poke2_recorded)
        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1.0  # no action
            r1 = 0
            events1 = []

        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            if self.timeout:
                events1 = []
            if "drink" in events1:
                self.ndrink1 += 1
            elif "poke" in events1:
                self.npoke1 += 1
            if self.water_available1 and "drink" in events1:
                r1 = 2.0
                self.ncorrect1 += 1
                self.gotwater1 = True
                # self.drink1_list.append(self.timesteps)
                self.freeze1 = np.random.randint(4)
                self.water_available1 = False
                if self.randomize:
                    self.water_coords1 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
                    self.poke_coords1 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
            elif (not self.water_available1) and "poke" in events1:
                r1 = 2.0
                self.water_available1 = True
                # self.poke1_list.append(self.timesteps)

            # elif "no_action" in events1:
            #     r1 = -0.1
            else:
                r1 = self.movement_reward

        if self.freeze2 > 0:
            self.freeze2 = self.freeze2 - 1.0  # no action
            r2 = 0
            events2 = []
        else:
            events2 = self._move(self.agent2_pos, action["agent2"], is_agent1=0)

            if self.timeout:
                events2 = []
            if "drink" in events2:
                self.ndrink2 += 1
            elif "poke" in events2:
                self.npoke2 += 1
            if self.water_available2 and "drink" in events2:
                r2 = 2.0
                self.ncorrect2 += 1
                self.gotwater2 = True
                # self.drink2_list.append(self.timesteps)
                self.freeze2 = np.random.randint(4)
                self.water_available2 = False
                if self.randomize:
                    self.water_coords2 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]  # upper left corner
                    self.poke_coords2 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]  # upper left corner
            elif (not self.water_available2) and "poke" in events2:
                r2 = 2.0
                self.water_available2 = True
                # self.poke2_list.append(self.timesteps)
            # elif "no_action" in events2:
            #     r2 = -0.1
            else:
                r2 = self.movement_reward

        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1
            self.ncorrect += 1
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both and create time out - any poking in the next time step does not count
        elif np.any(poke_mat[:, 0]):
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            self.miss = 1
            if poke_mat[0, 0]:
                self.nmiss1 += 1
            elif poke_mat[1, 0]:
                self.nmiss2 += 1
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }

        info = {
            "agent1": {
                "ncorrect": self.ncorrect,
                "ncorrect1": self.ncorrect1,
                "nmiss": self.nmiss1,
                "npoke1": self.npoke1,
                "ndrink1": self.ndrink1,
            },
            "agent2": {
                "ncorrect": self.ncorrect,
                "ncorrect2": self.ncorrect2,
                "nmiss": self.nmiss2,
                "npoke2": self.npoke2,
                "ndrink2": self.ndrink2,
            },
        }
        return obs, rewards, dones, info  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        # discrete coordinate of the locations of nose poke water port and
        return {
            "agent1": {
                "nosepoke0": self.poke_coords1[0],
                "water0": self.water_coords1[0],
                "self0": self.agent1_pos[0],
                # "otheragent0": self.agent2_pos[0],
                # "otherpoke0": self.poke_coords2[0],
                "nosepoke1": self.poke_coords1[1],
                "water1": self.water_coords1[1],
                "self1": self.agent1_pos[1],
                # "otheragent1": self.agent2_pos[1] - 4,
                # "otherpoke1": self.poke_coords2[1] - 4,
            },
            "agent2": {
                "nosepoke0": self.poke_coords2[0],
                "water0": self.water_coords2[0],
                "self0": self.agent2_pos[0],
                # "otheragent0": self.agent1_pos[0],
                # "otherpoke0": self.poke_coords1[0],
                "nosepoke1": self.poke_coords2[1] - 4,
                "water1": self.water_coords2[1] - 4,
                "self1": self.agent2_pos[1] - 4,
                # "otheragent1": self.agent1_pos[1],
                # "otherpoke1": self.poke_coords1[1],
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # check walls.

        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1

        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1

        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # update agent location
        # nose poke
        if is_agent1:
            if coords == self.poke_coords1:
                if not self.poke1_recorded:
                    self.poke1_recorded = True
                    return {"poke"}
            else:
                self.poke1_recorded = False

            if coords == self.water_coords1:
                return {"drink"}
        else:
            if coords == self.poke_coords2:
                if not self.poke2_recorded:
                    self.poke2_recorded = True
                    return {"poke"}
            else:
                self.poke2_recorded = False

            if coords == self.water_coords2:
                return {"drink"}

        # no action
        if action == 4:
            return {"no_action"}
        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

    def render_image(self):
        # Set up the image size and grid parameters
        image_size = (400, 400)
        increment = 25
        grid_size = (8, 8)
        agent1_area_width = 4

        # Create a blank white image
        image = Image.new("RGB", (image_size[0] + 50, image_size[1] + 50), "white")
        draw = ImageDraw.Draw(image)

        # Calculate cell size
        cell_size = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

        # Draw grid
        for i in range(grid_size[0] + 1):
            draw.line(
                [
                    (i * cell_size[0] + increment, 0),
                    (i * cell_size[0] + increment, image_size[1]),
                ],
                fill="black",
            )

        for j in range(grid_size[1] + 1):
            draw.line(
                [
                    (0 + increment, j * cell_size[1]),
                    (image_size[0] + increment, j * cell_size[1]),
                ],
                fill="black",
            )

        # Highlight agent areas
        agent1_area = (
            0 + increment,
            0,
            agent1_area_width * cell_size[0] + increment,
            image_size[1],
        )
        agent2_area = (
            agent1_area_width * cell_size[0] + increment,
            0,
            image_size[0] + increment,
            image_size[1],
        )

        draw.rectangle(agent1_area, outline="black", width=2)
        draw.rectangle(agent2_area, outline="black", width=2)

        # Add text information
        fontpath = "/home/kumquat/Downloads/aileron_0102/Aileron-Bold.otf"
        new_font_size = 20  # Adjust the size as needed
        font = ImageFont.truetype(fontpath, new_font_size)

        text_y_position = image_size[1] + 25

        text = (
            f"!SyncPoke!" if self.sync_poke == 1 else "!Miss!" if self.miss == 1 else ""
        )
        draw.text((170, text_y_position - 10), text, font=font, fill="black")
        # Move text information to the bottom
        self.draw_event_text(
            draw, self.events["agent1"], (30, text_y_position), font, "red"
        )
        self.draw_event_text(
            draw, self.events["agent2"], (290, text_y_position), font, "blue"
        )

        # Draw "o" at nose poke coordinates
        self.draw_symbol(
            draw, "N", font, self.poke_coords1, cell_size, increment, "yellow"
        )
        self.draw_symbol(
            draw, "N", font, self.poke_coords2, cell_size, increment, "yellow"
        )

        # Draw star at water coordinates
        self.draw_symbol(
            draw, "W", font, self.water_coords1, cell_size, increment, "green"
        )
        self.draw_symbol(
            draw, "W", font, self.water_coords2, cell_size, increment, "green"
        )
        # Draw agent positions
        self.draw_agent_position(
            draw, self.agent1_pos, increment, cell_size, color="red"
        )
        self.draw_agent_position(
            draw, self.agent2_pos, increment, cell_size, color="blue"
        )
        # Display the updated image
        self.image_list.append(image)
        # If you want to save the image, you can use self.image.save("filename.png")

    def draw_symbol(self, draw, symbol, font, position, cell_size, increment, color):
        # Calculate the center of the cell
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]
        left = position[1] * cell_size[0] + increment
        top = position[0] * cell_size[1]
        right = left + cell_size[0]
        bottom = top + cell_size[1]
        # paint the tile
        draw.rectangle([left, top, right, bottom], fill=color)
        # Draw the symbol at the center
        draw.text((center_x, center_y), symbol, font=font, fill="black", anchor="mm")

    def draw_agent_position(self, draw, position, increment, cell_size, color):
        center_x = (position[1] + 0.5) * cell_size[0] + increment
        center_y = (position[0] + 0.5) * cell_size[1]
        radius = (
            min(cell_size) / 4
        )  # Calculate radius as half of the minimum cell dimension

        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=color,
        )

    def draw_event_text(self, draw, events, position, font, color):
        text = (
            "!Agent Poke!"
            if "poke" in events
            else "!Agent Drink!" if "drink" in events else ""
        )
        draw.text(position, text, font=font, fill=color)
