def solve_sudoku(board):
    if is_complete(board):
        return board

    row, col = most_constraining_variable(board)
    available_nums = get_available_nums(board, row, col)

    # Apply heuristic ordering
    ordered_nums = least_constraining_value(board, row, col, available_nums)

    for num in ordered_nums:
        board[row][col] = num

        if solve_sudoku(board):
            return board

        board[row][col] = 0

    return None


def is_complete(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return False
    return True


def find_empty_cell(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None


def is_valid(board, row, col, num):
    # Check row
    for i in range(9):
        if board[row][i] == num:
            return False

    # Check column
    for i in range(9):
        if board[i][col] == num:
            return False

    # Check 3x3 subgrid
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True


def get_available_nums(board, row, col):
    used_nums = set()

    # Check row
    for i in range(9):
        if board[row][i] != 0:
            used_nums.add(board[row][i])

    # Check column
    for i in range(9):
        if board[i][col] != 0:
            used_nums.add(board[i][col])

    # Check 3x3 subgrid
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] != 0:
                used_nums.add(board[start_row + i][start_col + j])

    available_nums = []
    for num in range(1, 10):
        if num not in used_nums:
            available_nums.append(num)

    return available_nums


def most_constrained_variable(board):
    min_domain_size = float('inf')
    min_row = -1
    min_col = -1

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                available_nums = get_available_nums(board, row, col)
                domain_size = len(available_nums)

                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    min_row = row
                    min_col = col

    return min_row, min_col


def most_constraining_variable(board):
    max_constraints = -1
    max_row = -1
    max_col = -1

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                constraints = count_constraints(board, row, col)

                if constraints > max_constraints:
                    max_constraints = constraints
                    max_row = row
                    max_col = col

    return max_row, max_col


def least_constraining_value(board, row, col, nums):
    value_scores = []

    for num in nums:
        score = count_constraints(board, row, col, num)
        value_scores.append((num, score))

    value_scores.sort(key=lambda x: x[1])

    ordered_nums = [num for num, _ in value_scores]

    return ordered_nums


def count_constraints(board, row, col, num=None):
    constraints = 0

    for i in range(9):
        if board[row][i] == 0 and (num is None or num in get_available_nums(board, row, i)):
            constraints += 1

        if board[i][col] == 0 and (num is None or num in get_available_nums(board, i, col)):
            constraints += 1

    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == 0 and (num is None or num in get_available_nums(board, start_row + i, start_col + j)):
                constraints += 1

    return constraints


# Example Sudoku problem
board = [
    [0, 1, 0, 0, 0, 0, 0, 0, 6],
    [9, 0, 0, 2, 0, 0, 0, 0, 0],
    [7, 3, 2, 0, 4, 0, 0, 1, 0],
    [0, 4, 8, 3, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 4, 6, 7, 0],
    [0, 9, 0, 0, 3, 0, 5, 6, 8],
    [0, 0, 0, 0, 0, 2, 0, 0, 1],
    [6, 0, 0, 0, 0, 0, 0, 3, 0]
]

# Apply heuristic ordering
start_row, start_col = most_constrained_variable(board)
board = solve_sudoku(board)

if board:
    print("Solution to Sudoku:")
    for row in board:
        print(row)
else:
    print("Unable to solve the Sudoku problem.")
