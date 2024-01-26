import random
import time


def solve_sudoku(puzzle):
    global node_count
    node_count += 1
    if is_complete(puzzle):
        return puzzle

    row, col = find_empty_cell(puzzle)

    numbers = list(range(1, 10))
    random.shuffle(numbers)  # random value order

    for num in numbers:
        if is_valid(puzzle, row, col, num):
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


def find_empty_cell(puzzle):
    empty_cells = []
    for row in range(9):
        for col in range(9):
            if puzzle[row][col] == 0:
                empty_cells.append((row, col))
    if empty_cells:
        random.shuffle(empty_cells)  # random variable order
        return empty_cells[0]
    else:
        return None


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
    result = solve_sudoku(puzzle)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time, end=" ")
    print(node_count)

if result:
    print("B Solution：")
    for row in result:
        print(row)
else:
    print("Unable to solve.")
