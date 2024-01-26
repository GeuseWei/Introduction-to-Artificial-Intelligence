import random
import time

def solve_sudoku(puzzle):
    global node_count
    node_count += 1
    if is_complete(puzzle):
        return puzzle

    row, col = select_variable(puzzle)
    available_nums = get_available_nums(puzzle, row, col)

    ordered_nums = least_constraining_value(puzzle, row, col, available_nums)

    for num in ordered_nums:
        puzzle[row][col] = num

        if solve_sudoku(puzzle):
            return puzzle

        puzzle[row][col] = 0

    return None


def is_complete(puzzle):
    for row in range(9):
        for col in range(9):
            if puzzle[row][col] == 0:
                return False
    return True


def is_valid(puzzle, row, col, num):
    # Check row
    for i in range(9):
        if puzzle[row][i] == num:
            return False

    # Check column
    for i in range(9):
        if puzzle[i][col] == num:
            return False

    # Check 3-by-3 boxes
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if puzzle[start_row + i][start_col + j] == num:
                return False

    return True


# Forward checking
def get_available_nums(puzzle, row, col):
    used_nums = set()

    # Check row
    for i in range(9):
        if puzzle[row][i] != 0:
            used_nums.add(puzzle[row][i])

    # Check column
    for i in range(9):
        if puzzle[i][col] != 0:
            used_nums.add(puzzle[i][col])

    # Check 3-by-3 boxes
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if puzzle[start_row + i][start_col + j] != 0:
                used_nums.add(puzzle[start_row + i][start_col + j])

    available_nums = []
    for num in range(1, 10):
        if num not in used_nums:
            available_nums.append(num)

    return available_nums


def select_variable(puzzle):
    min_domain_size = float('inf')
    min_row = -1
    min_col = -1

    for row in range(9):
        for col in range(9):
            # most constrained variable heuristic
            if puzzle[row][col] == 0:
                available_nums = get_available_nums(puzzle, row, col)
                domain_size = len(available_nums)

                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    min_row = row
                    min_col = col

                # most constraining variable heuristic(if the first heuristic results in a first-place tie)
                elif domain_size == min_domain_size:
                    constraints_current = count_constraints(puzzle, row, col)
                    constraints_min = count_constraints(puzzle, min_row, min_col)
                    if constraints_current > constraints_min:
                        min_row = row
                        min_col = col
                    # break remaining ties in the order of the variables at random
                    elif constraints_current == constraints_min:
                        choice_list = [[min_row, min_col], [row, col]]
                        return random.choice(choice_list)
    return min_row, min_col


# least constraining value heuristic
def least_constraining_value(puzzle, row, col, nums):
    value_scores = []

    for num in nums:
        score = count_constraints(puzzle, row, col, num)
        value_scores.append((num, score))

    # break remaining ties in the order of the values at random
    value_scores.sort(key=lambda x: (x[1], random.random()))

    ordered_nums = [num for num, _ in value_scores]

    return ordered_nums


def count_constraints(puzzle, row, col, num=None):
    constraints = 0

    for i in range(9):
        if puzzle[row][i] == 0 and (num is None or num in get_available_nums(puzzle, row, i)):
            constraints += 1

        if puzzle[i][col] == 0 and (num is None or num in get_available_nums(puzzle, i, col)):
            constraints += 1

    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if puzzle[start_row + i][start_col + j] == 0 and (num is None or num in get_available_nums(puzzle, start_row + i, start_col + j)):
                constraints += 1

    return constraints


for i in range(50):
    puzzle = [
        [0, 5, 8, 0, 6, 2, 1, 0, 0],
        [0, 0, 2, 7, 0, 0, 4, 0, 0],
        [0, 6, 7, 9, 0, 1, 2, 5, 0],
        [0, 8, 6, 3, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 6, 8, 9, 0],
        [0, 2, 9, 6, 0, 8, 7, 4, 0],
        [0, 0, 3, 0, 0, 4, 9, 0, 0],
        [0, 0, 5, 2, 9, 0, 3, 8, 0]
    ]
    node_count = 0
    start_time = time.time()
    start_row, start_col = select_variable(puzzle)
    result = solve_sudoku(puzzle)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time, end=" ")
    print(node_count)


if result:
    print("B+FC+H Solution:")
    for row in result:
        print(row)
else:
    print("Unable to solve.")


