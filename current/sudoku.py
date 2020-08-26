import time


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


def same_row(i, j): return i // 9 == j // 9


def same_col(i, j): return (i - j) % 9 == 0


def same_block(i, j): return i // 27 == j // 27 and (i % 9) // 3 == (j % 9) // 3


def dfs(a):
    i = a.find('0')
    if i == -1:
        # we have found a solution
        raise Exception(a)

    excluded_numbers = set()
    for j in range(81):
        if same_row(i, j) or same_col(i, j) or same_block(i, j):
            excluded_numbers.add(a[j])

    for m in '123456789':
        if m not in excluded_numbers:
            # At this point, m is a valid choice, so place it and recurse
            dfs(a[:i] + m + a[i + 1:])


def solve(squares_num_array):
    # print_grid(squares_num_array)

    if squares_num_array.count('0') >= 80:
        return None, None

    start = time.time()
    try:
        dfs(squares_num_array)
        print('No solution')
        return None, None
    except Exception as e:
        print('Solved! Check the image')
        return str(e), "Solved in %.4fs" % (time.time() - start)
