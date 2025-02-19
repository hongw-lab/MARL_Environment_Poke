# RNN with one fully connected layer

import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

import logging
from typing import Callable

torch, nn = try_import_torch()


class AnotherTorchRNNModel(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        rnn_hidden_size=256,
        l2_lambda=0,
        l2_lambda_inp=0,
        device="cpu",
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.rnn_hidden_size = model_config["custom_model_config"]["rnn_hidden_size"]
        self.l2_lambda = model_config["custom_model_config"]["l2_lambda"]
        self.l2_lambda_inp = model_config["custom_model_config"]["l2_lambda_inp"]
        # self.rate_lambda = model_config["custom_model_config"]["rate_lambda"]
        self.noise_std = model_config["custom_model_config"]["noise_std"]
        # Build the Module from fc + RNN + 2xfc (action + value outs).
        # self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.rnn = nn.RNN(
            self.obs_size, self.rnn_hidden_size, batch_first=True, nonlinearity="relu"
        )
        self.action_branch = nn.Linear(self.rnn_hidden_size, num_outputs)
        self.value_branch = nn.Linear(self.rnn_hidden_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

        self.l2_loss = None
        self.l2_loss_inp = None
        self.original_loss = None

        self.activations = {}
        self.inputs = {}
        self.hooks = []
        self.device = device

        pretrained_weights = model_config["custom_model_config"].get(
            "pretrained_weights", None
        )
        #        pdb.set_trace()
        if pretrained_weights is not None:
            print("loading pretrained weights")
            self.load_pretrained_weights(pretrained_weights)

    def load_pretrained_weights(self, pretrained_weights):
        # Load the weights into the model
        model_dict = self.state_dict()
        for key in pretrained_weights:
            if key in model_dict:
                if isinstance(pretrained_weights[key], np.ndarray):
                    pretrained_weights[key] = torch.from_numpy(pretrained_weights[key])
        pretrained_weights = {
            k: v for k, v in pretrained_weights.items() if k in model_dict
        }
        model_dict.update(pretrained_weights)
        self.load_state_dict(model_dict)

    def register_activation_hooks(self):
        """
        Adds hooks to save activations from all neural network layers in this class.
        For more details on how this works, see:
        https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
        https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
        :return:
        """
        layer_names = set([name.split(".")[0] for name, _ in self.named_parameters()])
        self.inputs = {"rnn": {"inp": [], "rec": []}}

        # This yields a method that adds an activation to our dictionary.
        def save_activations_for(name) -> Callable:
            def save(model, input, output):
                # RNNs will output a tuple containing the same tensor twice, so we need to pick one.
                if type(model).__name__ == "RNN":
                    self.activations[name].append(output[0].cpu().detach())
                    self.inputs[name]["inp"].append(input[0].cpu().detach())
                    self.inputs[name]["rec"].append(input[1].cpu().detach())
                else:
                    self.activations[name].append(output.cpu().detach())

            return save

        for name in layer_names:
            layer = getattr(self, name)
            self.activations[name] = []  # Create activations list
            # We access the internal model of the RLlib module here. Not great, but it's the only thing that works.
            # hook = layer._model.register_forward_hook(save_activations_for(name))
            hook = layer.register_forward_hook(save_activations_for(name))
            self.hooks.append(hook)

        # also append input and two sets of weights - ih and hh:

    def deregister_activation_hooks(self):
        """
        Removes stored activation hooks
        :return:
        """
        for hook in self.hooks:
            hook.remove()

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.rnn.weight_ih_l0.new(1, self.rnn_hidden_size).zero_().squeeze(0),
            self.rnn.weight_ih_l0.new(1, self.rnn_hidden_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = inputs  # nn.functional.relu(self.fc1(inputs))
        y = [
            torch.unsqueeze(state[0], 0)
        ]  # y = [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        # print(y[0])
        # y[0] = y[0] + torch.randn_like(y[0]) * self.noise_std
        self._features, hn = self.rnn(x, y[0])
        # pdb.set_trace()
        action_out = self.action_branch(self._features)

        return action_out, [
            torch.squeeze(hn, 0)
        ]  # [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):

        l2_lambda = self.l2_lambda
        l2_reg = torch.tensor(0.0).to(self.device)
        # l2_reg += torch.norm(self.rnn.weight_hh_l0.data)
        l2_reg += torch.norm(self.rnn.weight_hh_l0).to(self.device)

        l2_lambda_inp = self.l2_lambda_inp
        l2_reg_inp = torch.tensor(0.0).to(self.device)
        l2_reg_inp += torch.norm(self.rnn.weight_ih_l0).to(self.device)

        self.l2_loss = l2_lambda * l2_reg
        self.l2_loss_inp = l2_lambda_inp * l2_reg_inp

        self.original_loss = policy_loss

        assert self.l2_loss.requires_grad, "l2 loss no gradient"
        assert self.l2_loss_inp.requires_grad, "l2 loss no gradient"

        custom_loss = self.l2_loss + self.l2_loss_inp


        total_loss = [p_loss + custom_loss for p_loss in policy_loss]

        return total_loss

    def metrics(self):
        metrics = {
            "weight_loss": self.l2_loss.item(),
            "original_loss": self.original_loss[0].item(),
        }
        # you can print them to command line here. with Torch models its somehow not reportet to the logger
        # print(metrics)

    def get_weights(self):
        weights = {"ih": self.rnn.weight_ih_l0, "hh": self.rnn.weight_hh_l0}
        return weights

    def set_weights(self, weights):
        # Reshape the flattened weights to their original shapes
        # weights.to(self.device)
        start = 0
        self.rnn.weight_ih_l0 = torch.nn.Parameter(weights["ih"].to(self.device))
        self.rnn.weight_hh_l0 = torch.nn.Parameter(weights["hh"].to(self.device))
