# semantic-loss-pytorch

Create `Semantic Loss` equivalent circuits in `PyTorch` using `SDDs` for knowledge compilation.


# Table of contents
**[1. Quick start](#faststart)**

**[2. Documentation](#doc)**

  * [2.1. Convert constraint to DIMACS](#tocnf)
  * [2.2. PySDD](#pysdd)
  * [2.3. Semantic losses](#semloss)

**[3. Credits](#credits)**

<a name="faststart"></a>
## Quick start

- Install this package
```bash
pip install git+https://github.com/lucadiliello/semantic-loss-pytorch.git
```

- Write your constraints respecting the `sympy` sintax, with variables like `X1.2` and operators like `And(X0.2.3, X1.1.1)`. All lines are put in `and` relationship. Convert to DIMACS syntax with:
```bash
python -m semantic_loss_pytorch.constraints_to_cnf.py -i <myinputfile>.sympy -o <dimacs>.txt 
```

- Install `PySDD` (`pip install PySDD` seems to give some errors):
```bash
pip install git+https://github.com/wannesm/PySDD.git
```

- Compile your constraint to a `vtree` and an `sdd` file. To do so, run:
```bash
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd
```
`PySDD` is only needed for this step. If you don't need to convert other 
dimacs constraints to `vtree` and `sdd` files, you can uninstall it.

- Use the semantic loss in your `PyTorch` project
```python
from semantic_loss_pytorch import SemanticLoss

batch_size = 8
# constraints over a 2x2 variable
x = torch.rand((batch_size, 2, 2))

sl = SemanticLoss('constraint.sdd', 'constraint.vtree')
loss, wmc, wmc_per_sample = sl(probabilities=x, output_wmc=True, output_wmc_per_sample=True)

# or maybe a simpler
loss = sl(probabilities=x)
# or if you are working with logits
loss = sl(logits=logits)
```

<a name="doc"></a>
## Documentation

<a name="tocnf"></a>  
### Convert constraint to DIMACS

`semantic_loss_pytorch.constraints_to_cnf` is a module which allows the writing
of constraints in propositional logic in the syntax of **sympy, and
then translate them to DIMACS**. Constraints are expressed
1 by line, and are considered to be in an `and` relationsip.

Moreover, it allows to refer to variables not just by a single
index, like X<sub>i</sub>, but via more indexes, X<sub>i.j.z</sub>, etc, as if
they were in a tensor of arbitrary shape.  
When this syntax
is translated to DIMACS, the indexes are converted in a single
dimension, as if the variables were in a mono dimensional
vector, while parsing, multi dimensional indexes must
respect the input shape (can't refer to variables that do
not exist, out of bounds, etc.)

By using the **sympy syntax**, constraints can now
be written with more operators:
- and (&) and or (|)
- Xor
- Nand
- Nor
- ITE, if then else
- implies, by using ">>" and "<<".
- Equivalent(X1, X2, X3), etc.
- check out https://docs.sympy.org/latest/modules/logic.html for
more alternatives to the syntax, like `Or(a, b)` instead of `a|b`.
- essentially, you are not limited to the syntax of the logic
module of sympy, but can access the whole package if you want to try
funky stuff, this is however not supported in this package and you will
probably meet unexpected behaviours, and you should
stick to logic operators. If you go looking for unexpected behaviour
you will find it.:shipit:

Example usage: let's say we have 4 variables with 3 possible
states (think of some multinomial distribution), we can imagine
our states as arranged in a tensor of shape [4,3]. We would
like to say that when the first variable assumes state 1, then
the second variable must assume state 2, moreover, the third variable
has always state 3.

Keep in mind that variables are referred starting from index 0.
```bash
# this is a comment
shape [4,3]

# i like blank lines


# my constraints

# var1.state1 implies var2.state2
X0.0 >> X.1.1

# var3 must always have state 3
X2.2

# given that states are mutually exclusive, we should also state that
X0.0 >> (~X0.1 & ~X0.2)
X0.1 >> (~X0.0 & ~X0.2)
X0.2 >> (~X0.0 & ~X0.1)

X1.0 >> (~X1.1 & ~X1.2)
X1.1 >> (~X1.0 & ~X1.2)
X1.2 >> (~X1.0 & ~X1.1)

X2.0 >> (~X2.1 & ~X2.2)
X2.1 >> (~X2.0 & ~X2.2)
X2.2 >> (~X2.0 & ~X2.1)

X3.0 >> (~X3.1 & ~X3.2)
X3.1 >> (~X3.0 & ~X3.2)
X3.2 >> (~X3.0 & ~X3.1)

# we should also state that variables must have at least 1 state
(X0.0 | X0.1 | X0.2)
(X1.0 | X1.1 | X1.2)
(X2.0 | X2.1 | X2.2)
(X3.0 | X3.1 | X3.2)
```

After writing this input file, you can simply call
the script.
```bash
python -m constraints_to_cnf.py -i myinputfile.txt -o dimacs.txt
```

The result would be the following DIMACS file:
```
c This file was generated with the constraints_to_cnf module in this project.
c Starting from file 'example.sympy'.
c There are 13 variables present in the constraints, and 12 total variables, given by the shape [4, 3].
c
p cnf 12 18
9 0
5 -1 0
1 2 3 0
4 5 6 0
7 8 9 0
10 11 12 0
-1 -2 0
-1 -3 0
-2 -3 0
-4 -5 0
-4 -6 0
-5 -6 0
-7 -8 0
-7 -9 0
-8 -9 0
-10 -11 0
-10 -12 0
-11 -12 0
```

###### Note that DIMACS refers to variables by starting from index 1, and not 0.

Note that `-p` is an optional argument to also specify
the number of processes to use while using sympy to parse
our constraints. This might be necessary if you have many constraints,
given that sympy seems to really take a hit when parsing long strings.
While parsing many constraints can be more or less helped by
adding processes, very long constraints on single lines
will slow down the process and it might be smarter to
put them to cnf and then set them 1 per line.

Note that you can omit the dot for the first index, for
better readability; meaning that X1.2 is equal to writing
X.1.2, or X1 is the same as X.1.


More complex shapes can be as easily used, i.e. [3,4,50,200,2] etc.,
finding use cases for this is left to the reader.

**caveat**: Evaluation from sympy is turned off during
parsing, meaning that you can write down constraints
that are False, like Equivalent(X0, ~X0), or having
X0 and ~X0 on different lines.
However, I have noticed that even without evaluation
there seems to be the chance of sympy evaluating
something directly to False, which would result
in having a single constraint in the DIMACS output file, "False".
However "False" is not part of the DIMACS syntax so it will
result in an error if you try to use this output with pysdd.

Tests are in the test directory (might take some time depending
on your computer).



<a name="pysdd"></a>  
### PySDD

To compile your DIMACS cnf files to vtrees and sdds, and to use make use of them
while running the main script you will need to install the `PySDD` module.  

You can install `PySDD` by calling:
```bash
pip install git+https://github.com/wannesm/PySDD.git
```

Compile the `dimacs` file with:
```bash
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd
```

`PySDD` will be used to build `sdd` + `vtree` files, that will be finally
used to create the equivalent `PyTorch` tree.



<a name="semloss"></a>  
### Semantic Loss

The semantic loss module will build a tree over the given tensor in such a way it will represent the formula encoded in the SDD.

It is a subclass of `torch.nn.modules.losses._Loss` and, when called, returns up to three tensors:
- `wmc_per_sample`: the weighted model count with respect to each given sample
- `wmc`: the average of `wmc_per_sample`
- `loss`: the negative logarithm of `wmc`
You must pass to the SL only one argument between `logits` and `probabilities`. `logits` are internally converted to
probabilities with a `sigmoid` layer. If you need a different normalization, do it yourself and feed the `probabilities` argument.

```python
import torch
from semantic_loss_pytorch import SemanticLoss

batch_size = 8
# constraints over a 2x2 variable
x = torch.rand((batch_size, 2, 2))

sl = SemanticLoss('constraint.sdd', 'constraint.vtree')

# now call the SL in different ways

# if `x` is probabilities
loss, wmc, wmc_per_sample = sl(probabilities=x, output_wmc=True, output_wmc_per_sample=True)
# if `x` is logits
loss, wmc, wmc_per_sample = sl(logits=x, output_wmc=True, output_wmc_per_sample=True)
# if you want only something
loss, wmc = sl(probabilities=x, output_wmc=True)
# or
loss, wmc_per_sample = sl(logits=x, output_wmc_per_sample=True)
# or maybe only
loss = sl(logits=x)
# or
loss = sl(probabilities=x)


loss.shape
# (1,)

wmc.shape
# (1,)

wmc_per_sample.shape
# (batch_size,)
```

<a name="credits"></a>
## Credits

Thanks to [Jacopo Gobbi](https://github.com/fruttasecca) for his ackowledgement in using pieces of his software!

Thanks to [PySDD](https://github.com/wannesm/PySDD) developers that made this possible!
