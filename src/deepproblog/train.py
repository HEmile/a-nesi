import signal
import time
import torch
from typing import List, Callable, Union, Optional

import storch
from deepproblog.dataset import DataLoader
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.sampling.sample import COST_NO_PROOF, COST_FOUND_PROOF
from deepproblog.semiring import Results
from deepproblog.utils.stop_condition import EpochStop
from deepproblog.utils.stop_condition import StopCondition
import wandb


class TrainObject(object):
    """
    An object that performs the training of the model and keeps track of the state of the training.
    """

    def __init__(self, model: Model):
        self.model = model
        self.accumulated_loss = 0
        self.IS_probability = 0
        self.i = 1
        self.start = 0
        self.prev_iter_time = 0
        self.epoch = 0
        self.previous_handler = None
        self.interrupt = False
        self.hooks = []
        self.timing = [0, 0, 0]

    def get_loss(self, batch: List[Query], backpropagate_loss: Callable) -> float:
        """
        Calculates and propagates the loss for a given batch of queries and loss function.
        :param batch: The batch of queries.
        :param backpropagate_loss:  The loss function. It should also perform the backpropagation.
        :return: The average loss over the batch
        """
        total_loss = 0
        result = self.model.solve(batch)
        for r in result:
            if r.ground_time:
                self.timing[0] += r.ground_time / len(batch)
            if r.compile_time:
                self.timing[1] += r.compile_time / len(batch)
            if r.eval_time:
                self.timing[2] += r.eval_time / len(batch)
        result = [
            (result[i], batch[i]) for i in range(len(batch)) if len(result[i]) > 0
        ]
        for r, q in result:
            total_loss += backpropagate_loss(
                r, q.p, weight=1 / len(result), q=q.substitute().query
            )
        return total_loss

    def get_loss_with_negatives(
            self, batch: List[Query], backpropagate_loss: Callable
    ) -> float:
        """
        Calculates and propagates the loss for a given batch of queries and loss function.
        This includes negative examples. Negative examples are found by using the query.replace_var method.
        :param batch: The batch of queries.
        :param backpropagate_loss:  The loss function. It should also perform the backpropagation.
        :return: The average loss over the batch
        """
        total_loss = 0

        result = self.model.solve([q.variable_output() for q in batch])
        result = [(result[i], batch[i]) for i in range(len(batch))]

        for r, q in result:
            expected = q.substitute().query
            try:
                total_loss += backpropagate_loss(
                    r, q.p, weight=1 / len(result), q=expected
                )
            except KeyError:
                self.get_loss([q], backpropagate_loss)
            neg_proofs = [x for x in r if x != expected]
            for neg in neg_proofs:
                # print('penalizing wrong answer {} vs {}'.format(q.substitute().query, k))
                total_loss += backpropagate_loss(
                    r, 0, weight=1 / (len(result) * len(neg_proofs)), q=neg
                )
        return total_loss

    def train(
            self,
            loader: DataLoader,
            stop_criterion: Union[int, StopCondition],
            args: dict,
            verbose: int = 1,
            loss_function_name: str = "cross_entropy",
            with_negatives: bool = False,
            log_iter: int = 100,
            initial_test: bool = True,
            **kwargs
    ) :

        self.previous_handler = signal.getsignal(signal.SIGINT)
        use_storch_training = False
        if hasattr(self.model.solver, "semiring"):
            loss_function = getattr(self.model.solver.semiring, loss_function_name)
        else:
            loss_function = None
            use_storch_training = True

        self.accumulated_loss = 0
        self.proof_prob = 0
        self.IS_probability = 0
        self.entropy = 0
        self.timing = [0, 0, 0]
        self.epoch = 0
        self.start = time.time()
        self.prev_iter_time = time.time()
        epoch_size = len(loader)
        # TODO: This is crashing on me, need to fix.
        # if "test" in kwargs and initial_test:
        #     value = kwargs["test"](self.model)
        #     wandb.log({"val": {"test": value}, "iteration": self.i, "epoch": self.epoch})
        #     print("Test: ", value)

        if type(stop_criterion) is int:
            stop_criterion = EpochStop(stop_criterion)
        print("Training ", stop_criterion)

        while not (stop_criterion.is_stop(self) or self.interrupt):
            epoch_start = time.time()
            self.model.optimizer.step_epoch()
            if verbose and epoch_size > log_iter:
                print("Epoch", self.epoch + 1)

            with torch.autograd.set_detect_anomaly(False):
                for batch in loader:
                    #     break
                    # while True:
                    if self.interrupt:
                        break
                    self.i += 1
                    self.model.train()
                    self.model.optimizer.zero_grad()
                    if use_storch_training:
                        # TODO: No loss for negatives
                        result = self.model.solve(batch)
                        assert isinstance(result, Results)
                        assert result.found_proof is not None

                        # # TODO: Can there ever be multiple keys? (ie multiple queries?
                        # #  Also: Should they then be combined like this?
                        # # Old code, we assume batching now
                        # if r.is_batched:
                        #     storch.add_cost(r.found_proof, 'found_proof_c')
                        #     self.accumulated_loss += storch.reduce_plates(r.found_proof.float())._tensor.data
                        #     # TODO: IS Probability
                        #     #  I don't know what I meant with this TODO.
                        #     self.IS_probability += torch.mean(r.found_proof._tensor.double())
                        # else:
                        #     for q in r.found_proof:
                        #         storch.add_cost(r.found_proof[q], f'found_proof_{q}')
                        #         self.accumulated_loss += torch.mean(r.found_proof[q]._tensor.double()) / len(result)
                        # Apply entropy minimization (positive) or maximization (negative)
                        # for s in r.stoch_tensors:
                        #     storch.add_cost(0.1*s.distribution.entropy(), f'entropy_{s.name}')

                        # assumes all costs are entropy
                        #     self.entropy += sum(map(lambda c: storch.reduce_plates(c)._tensor.data, storch.costs())) / (len(storch.costs()) * args["entropy_weight"])

                        storch.add_cost(-result.found_proof, 'found_proof_c')
                        iteration_prob = storch.reduce_plates(
                            result.found_proof / float(COST_FOUND_PROOF - COST_NO_PROOF) + 1/2
                        )._tensor.data
                        self.proof_prob += iteration_prob
                        self.accumulated_loss += storch.backward()
                        print(iteration_prob)
                    else:
                        if with_negatives:
                            loss = self.get_loss_with_negatives(batch, loss_function)
                        else:
                            loss = self.get_loss(batch, loss_function)
                        self.accumulated_loss += loss

                    self.model.optimizer.step()
                    self.log(verbose=verbose, log_iter=log_iter, **kwargs)
                    for j, hook in self.hooks:
                        if self.i % j == 0:
                            hook(self)

                    if stop_criterion.is_stop(self):
                        break
                if verbose and epoch_size > log_iter:
                    print("Epoch time: ", time.time() - epoch_start)
                self.epoch += 1
        if "snapshot_name" in kwargs:
            filename = "{}_final.mdl".format(kwargs["snapshot_name"])
            print("Writing snapshot to " + filename)
            self.model.save_state(filename)

        signal.signal(signal.SIGINT, self.previous_handler)

    def log(
            self, snapshot_iter: int = None, log_iter=100, test_iter=1000, verbose=1, **kwargs
    ):
        iter_time = time.time()

        if (
                "snapshot_name" in kwargs
                and snapshot_iter is not None
                and self.i % snapshot_iter == 0
        ):
            filename = "{}_iter_{}.mdl".format(kwargs["snapshot_name"], self.i)
            print("Writing snapshot to " + filename)
            self.model.save_state(filename)
        if self.i % log_iter == 0:
            to_log = {"iteration": self.i}
            print(
                "Iteration: ",
                self.i,
                "\ts:%.4f" % (iter_time - self.prev_iter_time),
                "\tAverage Loss: ",
                self.accumulated_loss / log_iter,
                "\tProof prob:",
                self.proof_prob / log_iter,
                "\tEntropy:",
                self.entropy / log_iter,
                flush=True
            )
            if len(self.model.parameters):
                print("\t".join(str(parameter) for parameter in self.model.parameters))
            to_log["train"] = {"loss": self.accumulated_loss / log_iter,
                               "proof_prob": self.proof_prob / log_iter,
                               "entropy": self.entropy / log_iter,
                               "ground_time": self.timing[0] / log_iter,
                               "compile_time": self.timing[1] / log_iter,
                               "eval_time": self.timing[2] / log_iter}
            # for k in self.model.parameters:
            #     self.logger.log(str(k), self.i, self.model.parameters[k])
            #     print(str(k), self.model.parameters[k])
            self.accumulated_loss = 0
            self.proof_prob = 0
            self.IS_probability = 0
            self.entropy = 0
            self.timing = [0, 0, 0]
            self.prev_iter_time = iter_time
            # TODO: Fix and re-enable. Apparently, wandb doesnt like sets, so make sure to fix that
            if "test" in kwargs and self.i % test_iter == 0:
                value = kwargs["test"](self.model)
                to_log["test"] = value
                print("Test: ", value)
            # print(to_log)
            wandb.log(to_log)

    def write_to_file(self, *args, **kwargs):
        print("Not implemented for wandb!")
        # self.logger.write_to_file(*args, **kwargs)


def train_model(
        model: Model,
        loader: DataLoader,
        stop_condition: Union[int, StopCondition],
        run_note: Optional[str] = None,
        run_tags: Optional[List[str]] = None,
        args: Optional[dict] = None,
        **kwargs
) -> TrainObject:
    args = args if args else {}
    wandb.init(
        project="deepproblog-mc",
        entity="hemile",
        notes=run_note,
        tags=run_tags,
        config={**model.get_hyperparameters(), **args}
    )
    train_object = TrainObject(model)
    train_object.train(loader, stop_condition, args, **kwargs)
    return train_object
