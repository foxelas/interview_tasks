WALL, PLAYER, BOX, GOAL, BOX_ON_GOAL, PLAYER_ON_GOAL, FLOOR = '#', '@', '$', '.', '*', '+', ' '

def parse_board(raw):
    lines = [line.rstrip('\n') for line in raw.splitlines() if line]
    width = max(len(line) for line in lines)
    board = [line.ljust(width) for line in lines]
    return board

def find_player(board):
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell in ('@', '+'):
                return x, y
    raise ValueError('Player not found')

def clone_board(board):
    return [list(row) for row in board]

def to_ascii(board):
    return '\n'.join(''.join(row) for row in board)

def to_structured(board):
    player = None
    boxes = []
    goals = []
    walls = []
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell == '#':
                walls.append([x, y])
            elif cell == '@':
                player = [x, y]
            elif cell == '+':
                player = [x, y]
                goals.append([x, y])
            elif cell == '$':
                boxes.append([x, y])
            elif cell == '*':
                boxes.append([x, y])
                goals.append([x, y])
            elif cell == '.':
                goals.append([x, y])
            # else: floor (space), do nothing
    return {
        "player": player,
        "boxes": boxes,
        "goals": goals,
        "walls": walls
    }


def to_(board, format_):
    if format_ == 'ascii':
        return to_ascii(board)
    elif format_ == 'structured':
        return to_structured(board)
    raise ValueError(f'Unknown format: {format_}')

def solved(board):
    for row in board:
        for cell in row:
            if cell == BOX:
                return False
    return True

def move(board, action):
    dx, dy = {'up': (0,-1), 'down': (0,1), 'left': (-1,0), 'right': (1,0)}[action]
    x, y = find_player(board)
    tx, ty = x+dx, y+dy
    max_y, max_x = len(board), len(board[0])
    if not (0 <= tx < max_x and 0 <= ty < max_y):
        return None  # out of bounds
    target = board[ty][tx]
    after = board[ty+dy][tx+dx] if 0 <= tx+dx < max_x and 0 <= ty+dy < max_y else None

    new_board = [list(row) for row in board]
    def set_cell(x, y, val): new_board[y][x] = val

    if target in (FLOOR, GOAL):
        # Move player into empty/goal
        if board[y][x] == PLAYER_ON_GOAL: set_cell(x, y, GOAL)
        else: set_cell(x, y, FLOOR)
        set_cell(tx, ty, PLAYER_ON_GOAL if target == GOAL else PLAYER)
        return new_board

    elif target in (BOX, BOX_ON_GOAL):
        if after not in (FLOOR, GOAL): return None  # can't push
        # Move box
        set_cell(tx+dx, ty+dy, BOX_ON_GOAL if after == GOAL else BOX)
        set_cell(tx, ty, PLAYER_ON_GOAL if target == BOX_ON_GOAL else PLAYER)
        set_cell(x, y, GOAL if board[y][x] == PLAYER_ON_GOAL else FLOOR)
        return new_board

    return None  # hit wall


class Sokoban:
    def __init__(self, ascii_board):
        self.initial = clone_board(parse_board(ascii_board))
        self.reset()
    def reset(self):
        self.board = clone_board(self.initial)
        self.moves = []
    def step(self, action):
        new_board = move(self.board, action)
        if new_board:
            self.board = new_board
            self.moves.append(action)
        return self
    def is_done(self):
        return solved(self.board)
    def render(self):
        print(to_ascii(self.board))