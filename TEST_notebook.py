# %%
import cv2
import numpy as np
from tetris import Tetris

# %%
game = Tetris()

test_moves = [
    (3, 0, 1),
    (5, 90, 2),
    (7, 180, 3), 
    (2, 270, 0),
]

for x_pos, angle, block_type in test_moves:
    game.current_piece = block_type
    game.current_angle = angle
    game.make_move(x_pos, angle, render=True, render_delay=0.5)

print("Final Board State:")
print(np.array(game.board))

cv2.waitKey(0)
cv2.destroyAllWindows()