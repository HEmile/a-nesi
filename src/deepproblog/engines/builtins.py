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


def tensor_wrapper(engine, func, *args):
    model = engine.model
    inputs = [to_tensor(model, a) for a in args]
    out = func(*inputs)
    return model.store_tensor(out)


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
