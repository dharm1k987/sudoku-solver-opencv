import time


def find_empty_cell(grid):
    for row in range(0, 9):
        for col in range(0, 9):
            if grid[row][col] == -1:
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

        grid[row][col] = -1

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


def solve(squares_num_array):
    print_grid(squares_num_array)
    start = time.time()
    if dfs(squares_num_array):
        print('Solved! Check the image')
        return squares_num_array, "Solved in %.4fs" % (time.time() - start)
    else:
        print('No solution')
        return None, None
