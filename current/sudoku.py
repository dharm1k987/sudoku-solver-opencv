import time
import numpy as np


def same_row(i,j): return (i/9 == j/9)
def same_col(i,j): return (i-j) % 9 == 0
def same_block(i,j): return (i/27 == j/27 and i%9/3 == j%9/3)

def r(a):
    i = a.find('0')
    if i == -1:
        return None, None

    excluded_numbers = set()
    for j in range(81):
        if same_row(i,j) or same_col(i,j) or same_block(i,j):
            excluded_numbers.add(a[j])

    for m in '123456789':
        if m not in excluded_numbers:
            # At this point, m is not excluded by any row, column, or block, so let's place it and recurse
            r(a[:i]+m+a[i+1:])






def find_empty_cell(grid):
    for row in range(0, 9):
        for col in range(0, 9):
            if grid[row][col] == 0:
                return row, col


def get_valid_choices(grid, row, col):
    result = []
    for x in range(1, 10):
        if no_conflict(grid, row, col, x):
            result.append(x)

    return result


def valid_hv(grid, row, col, choice):
    for x in range(0, 9):
        if grid[row][x] == choice or grid[x][col] == choice:
            return False

    return True


def valid_frame(grid, row, col, choice):
    for x in range(0, 3):
        for y in range(0, 3):
            if grid[row + x][col + y] == choice:
                return False

    return True


def no_conflict(grid, row, col, pick):
    if not valid_hv(grid, row, col, pick):
        return False
    if not valid_frame(grid, row - row % 3, col - col % 3, pick):
        return False
    return True


def dfs(grid):
    empty_cell = find_empty_cell(grid)

    if not empty_cell:
        return True

    row, col = empty_cell[0], empty_cell[1]

    stack = get_valid_choices(grid, row, col)

    while stack:
        choice = stack.pop()
        grid[row][col] = choice

        if dfs(grid):
            return True

        grid[row][col] = 0

    return False


def print_grid(grid):
    print("-" * 25)
    for i in range(9):
        print("|", end=" ")
        for j in range(9):
            print(grid[i][j], end=" ")
            if j % 3 == 2:
                print("|", end=" ")
        print()
        if i % 3 == 2:
            print("-" * 25)


def same_row(i,j): return (i//9 == j//9)
def same_col(i,j): return (i-j) % 9 == 0
def same_block(i,j): return (i//27 == j//27 and (i%9)//3 == (j%9)//3)

def r(a):
    i = a.find('0')
    if i == -1:
        raise Exception(a)

    excluded_numbers = set()
    for j in range(81):
        if same_row(i,j) or same_col(i,j) or same_block(i,j):
            excluded_numbers.add(a[j])

    for m in '123456789':
        if m not in excluded_numbers:
            # At this point, m is not excluded by any row, column, or block, so let's place it and recurse
            r(a[:i]+m+a[i+1:])



def solve(squares_num_array):
    # print_grid(squares_num_array)

    if squares_num_array.count('0') >= 80:
        return None, None

    start = time.time()
    try:
        r(squares_num_array)
        print('No solution')
        return None, None
    except Exception as e:
        print('Solved! Check the image')
        return str(e), "Solved in %.4fs" % (time.time() - start)









    if np.isclose(squares_num_array, 0).sum() == 81:
        return None, None
    copied_array = np.copy(squares_num_array)
    start = time.time()








    if dfs(copied_array):
        # print_grid(squares_num_array)
        print('Solved! Check the image')
        return copied_array, "Solved in %.4fs" % (time.time() - start)
    else:
        print('No solution')
        return None, None
