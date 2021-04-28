import numpy as np
import torch
import torch.nn.functional as F


class BasePolicy:
    """ An abstract class implementation of a policy function parametrization.
    The function takes as input the current state of the environment and
    returns a score for every action in the action space.
    """
    def __init__(self):
        raise NotImplementedError


    @torch.no_grad()
    def get_action(self, state, greedy=False):
        """ Return the action selected by the policy.
        Using the scores returned by the network compute a probability
        distribution over the actions from the action space. Select the next
        action probabilistically, or deterministically returning the action with
        the highest probability.

        @param state (np.ndarray): Numpy array of shape (batch_size, system_size),
                giving the current state of the environment.
        @param greedy (bool): If true, select the next action deterministically.
                If false, select the next action probabilistically.
        @return acts (np.ndarray): Numpy array of shape (batch_size,), giving
                actions selected by the policy for every state of the batch.
        """
        state = state.astype(np.float32)
        x = torch.from_numpy(state).to(self.device)
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        if greedy:
            acts = torch.argmax(probs, dim=1, keepdim=True)
        else:
            acts = torch.multinomial(probs, 1)
        acts = np.squeeze(acts.cpu().numpy(), axis=-1)
        return acts


    @property
    def device(self):
        """ Determine which device to place the Tensors upon, CPU or GPU. """
        return self.output_layer.weight.device


    @staticmethod
    def load(model_path):
        """ Load the model from a file. """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params["args"]
        model = FCNNPolicy(**args)
        model.load_state_dict(params["state_dict"])
        return model


    def save(self, path):
        """ Save the model to a file. """
        print("save model parameters to [%s]" % path)
        params = {"args": self.args,
                  "state_dict": self.state_dict()
                 }
        torch.save(params, path)

#