import time
import numpy as np

from itertools import product

'''
This solver was taken from https://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt under the GNU General Public License.
It expects input in the form of a 2D array, and will return the answer as a 2D array. If it is unsolvable, it will
raise an exception.
'''


def solve_sudoku(size, grid):
    R, C = size
    N = R * C
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])

    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // R) * R + (c // C)  # Box number
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = exact_cover(X, Y)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(X, Y, (i, j, n))
    for solution in solve(X, Y, []):
        for (r, c, n) in solution:
            grid[r][c] = n
        yield grid


def exact_cover(X, Y):
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y


def solve(X, Y, solution):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def solve_wrapper(squares_num_array):
    if squares_num_array.count('0') >= 80:
        return None, None

    start = time.time()

    # convert string to 9x9 array
    arr = []
    for i in squares_num_array:
        arr.append(int(i))

    arr = np.array(arr, dtype=int)
    arr = np.reshape(arr, (9, 9))
    try:
        ans = list(solve_sudoku(size=(3, 3), grid=arr))[0]
        s = ""
        for a in ans:
            s += "".join(str(x) for x in a)
        return s, "Solved in %.4fs" % (time.time() - start)
    except:
        return None, None
