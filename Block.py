import numpy as np
import copy

class Block():
    def __init__(self, block_matrix, shape):
        super().__init__()
        self.block_position = (1,3) 
        self.block_matrix = block_matrix # This is 2x2,3x3 or 4x4 depending from blocktype
        self.shape = shape # Shape of block
        self.timer = 1
        self.dropped = False
    
    def rotate_block(self, game_state):
        """
        Rotate Matrix 90 degrees clockwise
        """
        next_step_block = copy.deepcopy(self)
        next_step_block.block_matrix = self.rotate_matrix(next_step_block.block_matrix)
        if not game_state.check_collision(self, next_step_block):
            self.block_matrix = self.rotate_matrix(self.block_matrix)
    
    def rotate_matrix(self,m):
        return list(list(x)[::-1] for x in zip(*m))
    
    def move_left(self, game_state):
        """
        Move block left ones.
        """
        next_step_block = copy.deepcopy(self)
        next_step_block.block_position = (next_step_block.block_position[0], next_step_block.block_position[1] - 1)
        if not game_state.check_collision(self, next_step_block):
            self.block_position = (self.block_position[0], self.block_position[1] - 1)

    def move_right(self, game_state):
        """
        Move block right ones.
        """
        next_step_block = copy.deepcopy(self)
        next_step_block.block_position = (next_step_block.block_position[0], next_step_block.block_position[1] + 1)
        if not game_state.check_collision(self, next_step_block):
            self.block_position = (self.block_position[0], self.block_position[1] + 1)

    def move_down(self, game_state):
        """Move Block one space down
        
        Returns: True if able to move and False if not.
        """
        self.timer -= 1

        if self.timer <= 0 or self.dropped:
            next_step_block = copy.deepcopy(self)
            next_step_block.block_position = (next_step_block.block_position[0] + 1, next_step_block.block_position[1])
            if not game_state.check_collision(self, next_step_block):
                self.block_position = (self.block_position[0] + 1, self.block_position[1])
                self.timer = 1
                return True
            else:
                return False
        return True
    
    def calculate_height(self):
        height = 0
        for row in self.block_matrix:
            if len(set(row)) > 1 or self.set_unpacking(set(row)) > 0:
                height += 1
        return height
    
    def calculate_width(self):
        width = 0
        block_matrix = self.rotate_matrix(self.block_matrix)
        for row in block_matrix:
            if len(set(row)) > 1 or self.set_unpacking(set(row)) > 0:
                width += 1
        return width

    def set_unpacking(self, s):
        e, *_ = s
        return e