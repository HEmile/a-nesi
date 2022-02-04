from typing import Callable

import torch

from problog.logic import Term, Constant, is_list, term2list


def embed(engine, term):
    embedding = engine.model.get_embedding(term)[0, :]
    return Term("tensor", Constant(engine.tensor_store.store(embedding)))


def to_tensor(model, a):
    if type(a) is Term:
        if is_list(a):
            a = term2list(a)
        else:
            return model.get_tensor(a)
    # elif type(a) is Functor:
    #     return engine.tensor_store[int(a.args[0])]
    if type(a) is list:
        out = [to_tensor(model, x) for x in a]
        return [x for x in out if x is not None]
    else:
        return float(a)


def rbf(x, y):
    return torch.exp(-torch.norm(x - y, 2))


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def dot(x, y):
    return torch.dot(x, y)


def sigmoid(x):
    return torch.sigmoid(x)


def max(x):
    x = torch.stack(x, 0)
    x, _ = torch.max(x, 0)
    return x


def mean(x):
    x = torch.stack(x, 0)
    x = torch.mean(x, 0)
    return x


def one_hot(i, n):
    x = torch.zeros(int(n))
    x[int(i)] = 1.0
    return x


def cat(tensors):
    return torch.cat(tensors)


def stack(tensors):
    return torch.stack(tensors)

class _EmbedCall(Callable):
    def __init__(self, engine):
        self._engine = engine

    def __call__(self, *args):
        return embed(self._engine, *args)


class _WrappedCall(Callable):
    # Needed for pickl reasons
    def __init__(self, engine, func):
        self._engine = engine
        self._func = func

    def __call__(self, *args):
        model = self._engine.model
        inputs = [to_tensor(model, a) for a in args]
        out = self._func(*inputs)
        return model.store_tensor(out)


def register_tensor_predicates(engine):
    # Moved out of builtins.py for pickl reasons
    engine.register_foreign(_EmbedCall(engine), "embed", 1, 1)
    engine.register_foreign(_WrappedCall(engine, rbf), "rbf", 2, 1)
    engine.register_foreign(_WrappedCall(engine, add), "add", 2, 1)
    engine.register_foreign(_WrappedCall(engine, mul), "mul", 2, 1)
    engine.register_foreign(_WrappedCall(engine, dot), "dot", 2, 1)
    engine.register_foreign(_WrappedCall(engine, max), "max", 1, 1)
    engine.register_foreign(_WrappedCall(engine, sigmoid), "sigmoid", 1, 1)
    engine.register_foreign(_WrappedCall(engine, mean), "mean", 1, 1)
    engine.register_foreign(_WrappedCall(engine, stack), "stack", 1, 1)
    engine.register_foreign(_WrappedCall(engine, cat), "cat", 1, 1)
    engine.register_foreign(_WrappedCall(engine, one_hot), "one_hot", 2, 1)
