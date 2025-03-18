import random
import numpy as np
import cv2
from PIL import Image
from time import sleep

class Tetris():

    BOARD_WIDTH, BOARD_HEIGHT = 10, 20
    PLAYER_HOVER = 2
    MAP_EMPTY = 0
    # MAP_BLOCK = 1
    MAP_BLOCKS = { 
    0: 1,  # I 
    1: 2,  # T 
    2: 3,  # L 
    3: 4,  # J 
    4: 5,  # Z 
    5: 6,  # S 
    6: 7   # O 
    }
    MAP_COLORS = {
    0: (0, 0, 0),  # Empty (White)
    1: (224, 202, 0),    # I (Cyan)
    2: (225, 0, 189),    # T (Purple)
    3: (0, 158, 229),    # L (Orange)
    4: (229, 127, 2),    # J (Blue)
    5: (0, 0, 226),      # Z (Red)
    6: (57, 214, 0),     # S (Green)
    7: (1, 206, 223)     # O (Yellow)
    # COLORS = {
    # 'Black': (0, 0, 0),       # #000000 -> (0, 0, 0)
    # 'Green': (57, 214, 0),    # #00D639 -> (0, 214, 57) -> (57, 214, 0)
    # 'Purple': (225, 0, 189),  # #BD00E1 -> (189, 0, 225) -> (225, 0, 189)
    # 'Red': (0, 0, 226),       # #E20000 -> (226, 0, 0) -> (0, 0, 226)
    # 'Yellow': (1, 206, 223),  # #DFCE01 -> (223, 206, 1) -> (1, 206, 223)
    # 'Blue': (229, 127, 2),    # #027FE5 -> (2, 127, 229) -> (229, 127, 2)
    # 'Orange': (0, 158, 229),  # #E59E00 -> (229, 158, 0) -> (0, 158, 229)
    # 'Cyan': (224, 202, 0)     # #00CAE0 -> (0, 202, 224) -> (224, 202, 0) 
    # }
}

    BLOCKS = { # Taken from https://github.com/nuno-faria/tetris-ai/blob/master/tetris.py
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    # COLORS = {
    #     0: (255, 255, 255),
    #     1: (99, 64, 247),
    #     2: (247, 167, 0),
    # }


    def __init__(self):
        self.reset_board()

######################################## BLOCK FUNCTIONS ########################################


    def check_collision(self, piece, current_position):
        for x, y in piece:
            x += current_position[0]
            y += current_position[1]
            if (x < 0 or x >= Tetris.BOARD_WIDTH 
                    or y < 0 or y >= Tetris.BOARD_HEIGHT 
                    or self.board[y][x] > Tetris.MAP_EMPTY):
                return True
        return False
    

    def get_block_choice(self):
        if not self.block_list:
            self.block_list = list(self.BLOCKS.keys())
        return random.randint(0, len(self.block_list)-1) # NOTE: will likely have to change if you change color

    # Create a new piece: directly modifies the class variables
    def get_new_piece(self):
        random_choice = self.get_block_choice()

        self.current_piece = self.next_piece # We want to know next piece in advance to help inform the current move.
        self.next_piece = self.block_list.pop(random_choice) # This is the new piece we grab

        # self.color = Tetris.MAP_COLORS[Tetris.MAP_BLOCKS[self.current_piece]]
        self.current_angle = 0
        self.current_position = [3, 0] # NOTE: WHY?

        if self.check_collision(self.get_rotated_piece(), self.current_position): self.game_over = True   # NOTE: This function doesn't exist yet


    # Rotate piece: directly modifies the class variables
    def rotate_piece(self, angle_change):
        r = self.current_angle + angle_change
        
        # Make sure we are within bounds [0, 360)
        if r >= 360: r -= 360
        if r < 0: r += 360

        self.current_angle = r

    def get_rotated_piece(self): # NOTE: Can we combine this with above? does that make sense? let's see
        return Tetris.BLOCKS[self.current_piece][self.current_angle]


    # Place piece
    def place_piece(self, piece, position):
        current_board = [row[:] for row in self.board] # NOTE: I don't know what this does exactly yet
        for x, y in piece:
            current_board[y + position[1]][x + position[0]] = Tetris.MAP_BLOCKS[self.current_piece] # fill those coordinates  # NOTE: MAP_BLOCKS change
        return current_board

        
######################################## BOARD FUNCTIONS ########################################


    def get_num_empty_squares(self, board):
        board_arr = np.array(board)
        mask = board_arr > Tetris.MAP_EMPTY

        # Let's find the first block in each column. mask.argmax() will give us first occurence of True --> use argmax if the col is nonempty, else use board height
        first_block_ind = np.where(mask.any(axis=0), mask.argmax(axis=0), Tetris.BOARD_HEIGHT)
        # Sum of empty cells where the index is greater than (physically below) the first_block_ind
        return np.sum((board_arr == Tetris.MAP_EMPTY) & (np.arange(Tetris.BOARD_HEIGHT)[:, None] > first_block_ind))


    def get_bumpiness(self, board):
        board_arr = np.array(board)
        heights = np.argmax(board_arr > Tetris.MAP_EMPTY, axis = 0) # Find max index where there is a block. 

        heights[heights == 0] = Tetris.BOARD_HEIGHT # Bc we find max from the top, not from bottom
        bumpiness = np.abs(np.diff(heights)) # Get absolute value of difference between adjacent col heights
        return np.sum(bumpiness), np.max(bumpiness)


    def get_height(self, board):
        board_arr = np.array(board)
        heights = Tetris.BOARD_HEIGHT - np.argmax(board_arr > Tetris.MAP_EMPTY, axis=0) # we want actual height now, not just diffs
        heights[np.all(board_arr == Tetris.MAP_EMPTY, axis=0)] = 0 # NOTE: Should we replace these 2 lines in bumpiness with these?

        return np.sum(heights), np.max(heights), np.min(heights) 


    def get_board_properties(self, board): # To get the different stats to help train? i think?
        lines, board = self.clear_lines(board)
        holes = self.get_num_empty_squares(board) 
        total_bumpiness, max_bumpiness = self.get_bumpiness(board) # NOTE: max_bumpiness no longer needed?
        sum_height, max_height, min_height = self.get_height(board) # NOTE: max, min heights no longer needed?
        return [lines, holes, total_bumpiness, sum_height]

    
    def get_board(self):
        piece = self.get_rotated_piece()
        piece = [np.add(square, self.current_position) for square in piece]
        board = [x[:] for x in self.board]
        for x, y in piece: board[y][x] = Tetris.PLAYER_HOVER
        return board


    def clear_lines(self, board): # NOTE: why is board needed here? why can't we call self.board?
        cleared_lines = [col for col, row in enumerate(board) if all(x == 1 for x in row)] 
        if cleared_lines:
            board = [row for col, row in enumerate(board) if col not in cleared_lines] # NOTE: i think the col and row should be reverse..?
            # --> like we should probably be grabbing row for lines_to_clear right?
            for _ in cleared_lines: # Add new lines to top of board
                board.insert(0, [[0] for _ in Tetris.BOARD_WIDTH])
        return len(cleared_lines), board
            

    def reset_board(self):
        # Resets all of the states of the board
        self.game_over = False
        self.score = 0

        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]

        self.block_list = list(self.BLOCKS.keys())
        self.next_piece = self.block_list.pop(self.get_block_choice()) 
        self.get_new_piece()

        # Clear the board
        return self.get_board_properties(self.board) 
    

######################################## GAME FUNCTIONS ########################################

    def get_score(self):
        return self.score


    def make_move(self, x_pos, angle, render=False, render_delay=None):
        self.current_position = [x_pos, 0] # Find the block's x-axis position on the board
        self.current_angle = angle

        while not self.check_collision(self.get_rotated_piece(), self.current_position):
            if render:
                self.render_game()
                if render_delay: sleep(render_delay)
            self.current_position[1] += 1 # move block down until we collide with another block
        self.current_position[1] -= 1 # once we find a collision, we have to take 1 step backward

        self.board = self.place_piece(self.get_rotated_piece(), self.current_position)
        lines_cleared, self.board = self.clear_lines(self.board)
        score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH # Add 1 point for each piece placed plus reward for clearing lines
        self.score += score

        self.get_new_piece()

        if self.game_over:
            score -= 2  # NOTE: I assume this is a penalty for losing?
            self.render_game()  # Make sure final board is displayed NOTE: do we need this?

            cv2.waitKey(0)  # Wait indefinitely for user to press a key
        return score, self.game_over
    

    def render_game(self):
        board_state = self.get_board()
        image = [Tetris.MAP_COLORS[square_fill] for row in board_state for square_fill in row]
        image = np.array(image).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)

        moving_piece = self.get_rotated_piece()
        x_offset, y_offset = self.current_position

        for x, y in moving_piece:
            board_x, board_y = x + x_offset, y + y_offset
            if 0 <= board_x < Tetris.BOARD_WIDTH and 0 <= board_y < Tetris.BOARD_HEIGHT:
                image[board_y, board_x] = Tetris.MAP_COLORS[Tetris.MAP_BLOCKS[self.current_piece]]

        image = Image.fromarray(image, 'RGB')
        image = image.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        image = np.array(image)
        cv2.putText(image, str(self.score), (22, 22), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('image', np.array(image))
        cv2.waitKey(1)


######################################## MODEL FUNCTIONS ########################################
    
    def get_state_size(self): # State representation consists of this many elements/features:
        return 4
    

    def get_next_states(self): # Pass possible next states back to the model
        states = {}
        piece_id = self.current_piece
        
        # Set the rotations possible for each piece type
        if piece_id == 6: rotations = [0]
        elif piece_id == 0: rotations = [0, 90]
        else: rotations = [0, 90, 180, 270]

        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x): # For all positions
                pos = [x, 0]

                while not self.check_collision(piece, pos): pos[1] += 1 # Drop the piece
                pos[1] -= 1 # Backtrack 1 square once there's a collision

                if pos[1] >= 0: # Valid move (?)
                    board = self.place_piece(piece, pos)
                    states[(x, rotation)] = self.get_board_properties(board)

        return states

    
        