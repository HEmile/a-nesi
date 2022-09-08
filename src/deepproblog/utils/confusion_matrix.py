from typing import List, Union

import numpy as np

from deepproblog.utils import TabularFormatter
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class ConfusionMatrix(object):
    def __init__(self, classes: Union[int, List[str]] = 0):
        if isinstance(classes, int):
            self.n = classes
            self.classes = list(range(self.n))
        else:
            self.classes = classes
            self.n = len(classes)
        self.matrix = np.zeros((self.n, self.n), dtype=np.uint)

    def get_index(self, c):
        if c not in self.classes:
            self.grow(c)
        return self.classes.index(c)

    def grow(self, c):
        self.classes.append(c)
        self.n = len(self.classes)
        new_matrix = np.zeros((self.n, self.n), dtype=np.uint)
        new_matrix[0 : self.n - 1, 0 : self.n - 1] = self.matrix
        self.matrix = new_matrix

    def add_item(self, predicted, actual):
        actual_i = self.get_index(actual)
        predicted_i = self.get_index(predicted)

        self.matrix[predicted_i, actual_i] += 1

    def __str__(self, sort_by_class = True):
        formatter = TabularFormatter()
        permutation = sorted(range(len(self.classes)), key=lambda i: natural_keys(self.classes[i])) if sort_by_class else range(self.n)
        classes_s = [self.classes[i] for i in permutation]
        data = [[""] * (self.n + 2), ["", ""] + classes_s] + [] * self.n
        data[0][(self.n + 1) // 2 + 1] = "Actual"
        for row in permutation:
            data.append(
                [" ", classes_s[row]]
                + [str(self.matrix[row, col]) for col in permutation]
            )
        data[len(data) // 2][0] = "Predicted"
        return formatter.format(data)

    def accuracy(self):
        correct = 0
        for i in range(self.n):
            correct += self.matrix[i, i]
        total = self.matrix.sum()
        acc = correct / total
        print("Accuracy: ", acc)
        return acc
