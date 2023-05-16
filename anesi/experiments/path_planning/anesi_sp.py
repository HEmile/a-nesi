from typing import Optional, List, Tuple

import torch
from torch import Tensor, nn

from anesi import ANeSIBase
from experiments.path_planning.dijkstra import compute_shortest_path
from experiments.path_planning.im_pp import IMCNN, IMPath
from experiments.path_planning.perception import get_resnet, SPPerception, CombResnet18
from experiments.path_planning.state import SPState, SPStatePath, all_transport_y

from inference_models import ST, InferenceResult, NoPossibleActionsException


class SPModel(ANeSIBase[SPState], nn.Module):

    def __init__(self, args):
        self.N = args["N"]

        self.weight_types = args["weight_types"]
        self.unique_weights = len(self.weight_types)
        self.weight_types = torch.tensor(self.weight_types, dtype=torch.float32)

        self.use_path = args["q_model"].startswith("path")

        if self.use_path:
            im = IMPath(self.N, self.unique_weights, model=args["q_model"])
        else:
            im = IMCNN(self.N, self.unique_weights, model=args["q_model"])

        if args["perception_model"] == "comb_resnet":
            perception = CombResnet18(self.N, self.unique_weights)
        else:
            perception = SPPerception(self.N, self.unique_weights, model=args["perception_model"])

        super().__init__(im,
                         perception,
                         belief_size=[self.unique_weights] * (self.N * self.N),
                         **args)
        self.im = im
        self.verbosity = args['verbose']
        self.should_print = False

    def forward(self, preds: torch.Tensor):
        if self.training:
            return self.req_loss(preds)
        raise NotImplementedError()

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> SPState:
        w = [w] if w is not None else []
        if self.use_path:
            sy = []
            last_y1 = torch.zeros(P.shape[0], dtype=torch.long, device=P.device)
            last_y2 = torch.zeros(P.shape[0], dtype=torch.long, device=P.device)
            if y is not None:
                mask = (last_y1 != self.N * self.N - 1)
                # Creates a list of the transitions taken in the grid y
                while mask.any():
                    new_y = last_y1.clone()
                    # For finished paths, set the direction to 8 (no direction)
                    new_dir = 8 * torch.ones((new_y.shape[0],), device=P.device, dtype=torch.long)

                    # Gather all directions
                    all_transp_y = all_transport_y(last_y1[mask], self.N)
                    # Find the direction that is not the last direction but is a valid direction present in the path
                    # Uses multiplication for the elementwise-and
                    # Using torch.abs in the last line to set -1 values to 1 to prevent indexing errors.
                    # The -1 values will be filtered out in the first line
                    transp_mask = (all_transp_y != -1) * \
                                  (all_transp_y != last_y2[mask].unsqueeze(-1)) * \
                                  (torch.gather(y[mask], 1, torch.abs(all_transp_y)) == 1)
                    if (transp_mask.sum(-1) != 1).any():
                        index = (transp_mask.sum(-1) != 1).nonzero(as_tuple=True)[0]
                        print("Grid", y[mask][index].reshape(12, 12))
                        print("Last y1", last_y1[mask][index])
                        print("Last y2", last_y2[mask][index])
                        print("Transp mask", transp_mask[index])
                        print("Options", all_transp_y[index])
                        raise ValueError("Invalid path")

                    # Set the new direction and new y
                    new_y[mask] = all_transp_y[transp_mask]
                    new_dir[mask] = transp_mask.nonzero(as_tuple=True)[1]
                    last_y2 = last_y1
                    last_y1 = new_y
                    sy.append(new_dir)

                    mask = (last_y1 != self.N * self.N - 1)
                sy = [torch.stack(sy, dim=-1)]

            return SPStatePath(P, (sy, w), self.N, generate_w=generate_w)
        y = [y] if y is not None else []
        # Compute the inverse of torch.stack on w

        return SPState(P, (y, w), self.N, generate_w=generate_w)

    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (batch_size, k_vars)
        """
        batch_shape = w.shape[:-1]
        w_grid = w.reshape(-1, self.N, self.N)
        costs = self.weight_types.to(w.device)[w_grid]
        shortest_path = compute_shortest_path(costs).long().reshape(*batch_shape, self.N * self.N)
        if self.should_print:
            self._print_path(shortest_path[0], costs[0])
        return shortest_path

    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        return prediction == y

    def _print_path(self, path: torch.Tensor, cost: torch.Tensor):
        path = path.reshape(self.N, self.N)
        cost = cost.reshape(self.N, self.N)

        for x in range(self.N):
            for y in range(self.N):
                if path[x, y] == 0:
                    print(f" {cost[x, y]:.1f}  ", end=" ")
                else:
                    print(f"*{cost[x, y]:.1f}* ", end=" ")
            print()
        print()

    def cost_paths(self, path: torch.Tensor, cost: torch.Tensor):
        # paths: (batch_size, N * N) indicating the path taken
        # cost: (batch_size, N * N) indicating the cost of each cell
        # Returns: (batch_size,) indicating the cost of each path

        # Get the indices of the path
        path_costs = path * cost
        return path_costs.sum(-1)


    def test(self, x: torch.Tensor, y: torch.Tensor, true_w: Optional[List[torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Algorithm 1
        Sample from the PPE model
        :param x: The data to perform beam search on
        :param y: The true label y to predict and compare to
        :param true_w: The true w (explanation to get to y). Used to evaluate if explanations are correct
        """
        with torch.no_grad():
            P = self.perception(x)
            successes = torch.tensor(0, device=P.device)

            if not self.use_path:
                # TODO: We're getting errors because the beam search is not working properly and it can get to states
                #  with no actions possible.
                initial_state = self.initial_state(P, generate_w=False)
                try:
                    result: InferenceResult[ST] = self.q(initial_state, beam=self.use_path)
                    pred_y = (result.distributions[0]).max(dim=-1)[1]
                    successes = self.success(pred_y, y, beam=False).float().min(-1)[0].mean()
                except NoPossibleActionsException:
                    print("No possible actions during testing, skipping")
                    pass
                except RuntimeError as e:
                    print(e)
                    print("Runtime error during testing, skipping")
                    pass

            prior_predictions = torch.argmax(P, dim=-1)
            costs = self.weight_types.to(prior_predictions.device)[prior_predictions]
            w_accuracy = torch.isclose(costs, true_w, atol=0.01).float().mean()
            if self.verbosity > 0:
                self.should_print = True
                true_weights_1 = true_w[0]
                true_path_1 = y[0]
                print("True path and weights:")
                self._print_path(true_path_1, true_weights_1)
                print()
            prior_y = self.symbolic_function(prior_predictions)
            cost_prior = self.cost_paths(prior_y, true_w)
            cost_true = self.cost_paths(y, true_w)
            if self.verbosity < 2:
                self.should_print = False
            prior_acc = torch.isclose(cost_prior, cost_true, atol=0.1).float().mean()
            avg_dist = torch.abs(cost_prior - cost_true).mean()

            prior_acc_hamming = (prior_y == y).float().sum(-1).mean()

        # if true_w is not None:
        #     explain_acc = torch.tensor(0., device=successes.device)
        #     if not self.predict_only:
        #         for i in range(len(true_w)):
        #             # Get beam search prediction of w, compare to ground truth w
        #             explain_acc += (result.final_state.w[i][:, 0] == true_w[i]).float().mean()
        #         explain_acc /= len(true_w)
        #
        #     prior_acc = (prior_predictions == torch.stack(true_w, 1)).float().mean()
        #
        #     return torch.mean(successes), successes_prior, explain_acc, prior_acc
        return successes, prior_acc, prior_acc_hamming, w_accuracy, avg_dist

    def preprocess_y(self, y: torch.Tensor) -> List[torch.Tensor]:
        # TODO
        raise NotImplementedError()
