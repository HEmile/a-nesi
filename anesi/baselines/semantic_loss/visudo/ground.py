from __future__ import annotations

import math
from typing import List, Tuple

Literal = Tuple[int, bool]
Clause: List[Literal]

def encode(row: int, column: int, digit: int, N: int) -> int:
    """Encodes a sudoku cell as a number"""
    return row * N + column * N * N + digit
def ground_sudoku(N: int) -> List[Clause]:
    """Returns a propositional grounding for sudokus of size N x N"""
    exactly_one: List[Clause] = []
    mutually_exclusive_reflexive: List[Clause] = []

    for column in range(N):
        for row in range(N):
            # Each cell has at least one value
            exactly_one.append([(encode(row, column, digit, N), True) for digit in range(N)])

            # Each cell has at most one value
            for digit1 in range(N):
                for digit2 in range(digit1 + 1, N):
                    mutually_exclusive_reflexive.append([(encode(row, column, digit1, N), False),
                                                        (encode(row, column, digit2, N), False)])

    mutually_exclusive_pairs: List[Clause] = []
    for i in range(N * N):
        xi = i % N
        yi = i // N

        bxi = xi // math.sqrt(N)
        byi = yi // math.sqrt(N)
        for j in range(i + 1, N * N):
            xj = j % N
            yj = j // N

            bxj = xj // math.sqrt(N)
            byj = yj // math.sqrt(N)

            # Same row, same column, same box
            if xi == xj or yi == yj or (bxi == bxj and byi == byj):
                for digit in range(N):
                    mutually_exclusive_pairs.append([(encode(yi, xi, digit, N), False),
                                                    (encode(yj, xj, digit, N), False)])


    clauses = exactly_one + mutually_exclusive_reflexive + mutually_exclusive_pairs
    return clauses

def convert_sudoku(sudoku: List[List[int]]) -> List[bool]:
    """Converts a sudoku puzzle into a list of literals"""
    N = len(sudoku)
    literals = []
    for row in range(N):
        for column in range(N):
            digit = sudoku[row][column]
            if digit != 0:
                for _d2 in range(N):
                    if _d2 != digit - 1:
                        literals.append(False)
                    else:
                        literals.append(True)
    return literals


def test_clause(clause: Clause, literals: List[bool]) -> bool:
    return any(literals[literal[0]] == literal[1] for literal in clause)


def test_cnf(clauses: List[Clause], literals: List[bool]) -> bool:
    return all(test_clause(clause, literals) for clause in clauses)


def test_ground_sudoku():
    N = 9
    clauses = ground_sudoku(N)

    example_puzzle = [
        [4, 3, 5, 2, 6, 9, 7, 8, 1],
        [6, 8, 2, 5, 7, 1, 4, 9, 3],
        [1, 9, 7, 8, 3, 4, 5, 6, 2],
        [8, 2, 6, 1, 9, 5, 3, 4, 7],
        [3, 7, 4, 6, 8, 2, 9, 1, 5],
        [9, 5, 1, 7, 4, 3, 6, 2, 8],
        [5, 1, 9, 3, 2, 6, 8, 7, 4],
        [2, 4, 8, 9, 5, 7, 1, 3, 6],
        [7, 6, 3, 4, 1, 8, 2, 5, 9],
    ]

    literals = convert_sudoku(example_puzzle)
    assert len(literals) == 9 * 9 * 9
    assert test_cnf(clauses, literals)

    example_puzzle[4][7] = 2

    literals = convert_sudoku(example_puzzle)
    assert not test_cnf(clauses, literals)



if __name__ == "__main__":
    test_ground_sudoku()
    N = 4
    clauses = ground_sudoku(N)
    with open(f"sudoku_{N}.cnf", "w") as f:
        f.write(f"p cnf {N * N * N} {len(clauses)}\n")
        for clause in clauses:
            clause_cnf = " ".join([f"{'-' if not literal[1] else ''}{literal[0] + 1}" for literal in clause]) + " 0\n"
            f.write(clause_cnf)