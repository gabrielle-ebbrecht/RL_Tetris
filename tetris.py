import random
import numpy as np

class Tetris():
    # Things I probably need off the top of my head:
        # the actual grid structure
        # place a block - DONE
            # initialize a new piece - DONE
                # randomly get the next block - DONE
            # rotate block - DONE
        # clear a line
            # get the points from that line
        # game over condition
        # reset the game
            # start the game?
        # reward system
        # render the game so we can see as we play
    
    # constants usually written in all caps
    # nested dictionary with the 2D vals as if you're in the bottom left corner of the grid
        # layer 1 is the blocktype, layer 2 is the rotation

    BOARD_WIDTH, BOARD_HEIGHT = 10, 20
    PLAYER_HOVER = 2

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

    COLORS = {
        'Black': '#000000',
        'Green': '#00D639',
        'Purple': '#BD00E1',
        'Red': '#E20000', 
        'Yellow': '#DFCE01', 
        'Blue': '#027FE5', 
        'Orange': '#E59E00', 
        'Cyan': '#00CAE0'
    }

    def __init__(self):
        self.reset_board()

######################################## BLOCK FUNCTIONS ########################################


    def check_overflow(self, piece_details, current_position):
        # Is this needed for just regular gameplay to enforce edges?
        return None # filler
    

    def get_block_choice(self):
        if not self.block_list:
            self.block_list = self.BLOCKS.keys()
        return random.randint(0, len(self.BLOCKS))

    # Create a new piece: directly modifies the class variables
    def get_new_piece(self):
        random_choice = self.get_block_choice()

        self.current_piece = self.next_piece # We want to know next piece in advance to help inform the current move.
        self.next_piece = self.block_list.pop(random_choice) # This is the new piece we grab

        self.current_angle = 0
        self.current_position = [3, 0] # NOTE: WHY?

        if self.check_overflow(self.get_piece_details, self.current_position): self.game_over = True      


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
        current_board = [x[:] for x in self.board] # NOTE: I don't know what this does exactly yet
        for x, y in piece:
            # NOTE: maybe here is were we can assign color later with different values?
            current_board[y + position[1]][x + position[0]] = 1 # fill those coordinates 
        return current_board

        
######################################## BOARD FUNCTIONS ########################################


    def get_num_empty_squares(self, board):
        board_arr = np.array(board)
        mask = board_arr == 1

        # Let's find the first block in each column. mask.argmax() will give us first occurence of True --> use argmax if the col is nonempty, else use board height
        first_block_ind = np.where(mask.any(axis=0), mask.argmax(axis=0), Tetris.BOARD_HEIGHT)
        # Sum of empty cells where the index is greater than (physically below) the first_block_ind
        return np.sum(board_arr = 0) & np.arange(Tetris.BOARD_HEIGHT)[:, None] > first_block_ind


    def get_bumpiness(self, board):
        board_arr = np.array(board)
        heights = np.argmax(board_arr == 1, axis = 0) # Find max index where there is a 

        heights[heights == 0] = Tetris.BOARD_HEIGHT # Bc we find max from the top, not from bottom
        bumpiness = np.abs(np.diff(heights)) # Get absolute value of difference between adjacent col heights
        return np.sum(bumpiness), np.max(bumpiness)


    def get_height(self, board):
        board_arr = np.array(board)
        heights = Tetris.BOARD_HEIGHT - np.argmax(board_arr != 0, axis=0) # we want actual height now, not just diffs
        heights[np.all(board_arr == 0, axis=0)] = 0 # NOTE: Should we replace these 2 lines in bumpiness with these?

        return np.sum(heights), np.max(heights), np.min(heights) 


    def get_board_properties(self, board): # To get the different stats to help train? i think?
        lines, board = self.clear_lines(board)
        holes = self.get_num_empty_squares(board) 
        total_bumpiness, max_bumpiness = self.get_bumpiness(board) # NOTE: max_bumpiness no longer needed?
        sum_height, max_height, min_height = self.get_height(board) # NOTE: max, min heights no longer needed?
        return [lines, holes, total_bumpiness, sum_height]

    
    def get_board(self):
        piece = self.get_rotated_piece
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
                board.insert(0, [[0] for _ in Tetris.BOARD_WIDTH]) # NOTE: should it be Tetris.BOARD_WIDTH? Why? --> I think so bc it's a constant not a function
        return len(cleared_lines), board
            

    def reset_board(self):
        # Resets all of the states of the board
        self.game_over = False
        self.score = 0
        self.get_new_piece()

        self.board = [[0] * Tetris.BOARD_WIDTH for _ in Tetris.BOARD_HEIGHT]

        self.block_list = self.BLOCKS.keys()
        self.next_piece = self.block_list.pop(self.get_block_choice()) 

        # Clear the board
        return self.get_board_properties(self.board) 
    

######################################## GAME FUNCTIONS ########################################

    def get_score(self):
        return self.score


    def play_tetris(self):
        x = None
    
    
    def render_game(self):
        x = None
        