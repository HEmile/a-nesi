import pytest
import torch
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from semantic_loss_pytorch import SemanticLoss


# Test parameters are parsed correctly
@pytest.mark.parametrize("test_input, expected_outputs", [
    ([0.1, 0.2, 0.4, 0.3], [0.3]),
    ([0.6, 0.2, 0.1, 0.1], [0.5]),
])
def test_parser(test_input, expected_outputs):
    # create these files with constraints_to_cnf.py + pysdd
    sl = SemanticLoss('constraint.sdd', 'constraint.vtree')

    input_ = torch.tensor(test_input).unsqueeze(0) # add batch size dim even if it is 1

    assert sl(input_) == torch.tensor(expected_outputs)


