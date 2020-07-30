import pygame
import os
import copy
import numpy as np

pygame.font.init()

class GameState():
    def __init__(self):
        self.game_state =  [[0] * 10 for i in range(22)]

    def draw_window(self, win, current_block, score, next_block):

        ## We need to parameters game state 
        ## From gamestart we draw corrent blocks
        ## Game state will be 22x10 list of lists
        ## Draw Current Block on game grid after checking collision

        self.drawGrid(win)
        self.draw_blocks(win, current_block)
        self.draw_next_block(win, next_block)

        pygame.display.update()

    def drawGrid(self, win):
        """
        Draw grid for tetris game (22x10)
        """
        start = (70, -60)
        end = (370, 600)

        blockSize = 29 #Set the size of the grid block
        for x in range(start[0],end[0], 30):
            for y in range(start[1],end[1], 30):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(win, (230, 230, 230, 10), rect, 1)
        
        # Draw outside frame
        frame_rect = pygame.Rect(start[0] - 1, start[1]- 1, 301, 661)
        pygame.draw.rect(win, (100, 100, 100, 40), frame_rect, 1)
    
    def draw_next_block(self, win, next_block):
        """
        Draw next block hint.
        """

        # Position of next block window
        start = (450, 90)
        end = (570, 210)

        # Draw outside frame
        frame_rect = pygame.Rect(start[0] - 25, start[1] - 25, end[0]- 400, end[1] - 100)
        pygame.draw.rect(win, (100, 100, 100, 40), frame_rect, 1)

        blockSize = 29 #Set the size of the grid block  
        block_len = len(next_block.block_matrix)

        # change start position in block length 2 or 3 to set it middle
        if block_len == 3:
            start = (465, 90)
        elif block_len == 2:
            start = (480, 90)

        # Drow blocks to next block area
        for index_x, x in enumerate(range(start[0],end[0], 30)):
            for index_y, y in enumerate(range(start[1],end[1], 30)):
                if index_x < block_len and index_y < block_len:
                    value = next_block.block_matrix[index_y][index_x]
                    if value > 0:
                        rect = pygame.Rect(x, y, blockSize, blockSize)
                        if value == 1:
                            pygame.draw.rect(win, (230, 230, 0), rect, 0) ## Yellow
                        elif value == 2:
                            pygame.draw.rect(win, (230, 0, 0), rect, 0) ## Red
                        elif value == 3:
                            pygame.draw.rect(win, (0 ,230 , 0), rect, 0) ## Green
                        elif value == 4:
                            pygame.draw.rect(win, (0, 230, 230), rect, 0) ## Cyan
                        elif value == 5:
                            pygame.draw.rect(win, (230, 0, 230), rect, 0) ## Purple
                        elif value == 6:
                            pygame.draw.rect(win, (255, 125, 0), rect, 0) ## Orange
                        elif value == 7:
                            pygame.draw.rect(win, (0, 0, 230), rect, 0) ## Blue


    def draw_blocks(self, win, current_block):
        """
        Draw Block in grid
        """
        start = (70, -60)
        end = (370, 600)
        blockSize = 29 #Set the size of the grid block
        current_game_state = self.set_current_block_to_gamestate(current_block)
        for index_x, x in enumerate(range(start[0],end[0], 30)):
            for index_y, y in enumerate(range(start[1],end[1], 30)):
                game_state_value = current_game_state[index_y][index_x]
                if game_state_value > 0:
                    rect = pygame.Rect(x, y, blockSize, blockSize)
                    if game_state_value == 1:
                        pygame.draw.rect(win, (230, 230, 0), rect, 0) ## Yellow
                    elif game_state_value == 2:
                        pygame.draw.rect(win, (230, 0, 0), rect, 0) ## Red
                    elif game_state_value == 3:
                        pygame.draw.rect(win, (0 ,230 , 0), rect, 0) ## Green
                    elif game_state_value == 4:
                        pygame.draw.rect(win, (0, 230, 230), rect, 0) ## Cyan
                    elif game_state_value == 5:
                        pygame.draw.rect(win, (230, 0, 230), rect, 0) ## Purple
                    elif game_state_value == 6:
                        pygame.draw.rect(win, (255, 125, 0), rect, 0) ## Orange
                    elif game_state_value == 7:
                        pygame.draw.rect(win, (0, 0, 230), rect, 0) ## Blue

    def set_current_block_to_gamestate(self, current_block):
        """
        Append current block in current gamestate.

        Returns:
            New gamestate with current block in it.
        """
        current_game_state = copy.deepcopy(self.game_state)
        for x, row in enumerate(current_block.block_matrix):
            for y, e in enumerate(row):
                if e != 0 and x + current_block.block_position[0] < 22 and current_block.block_position[1] >= 0 - y and y + current_block.block_position[1] < 10:
                    current_game_state[x + current_block.block_position[0]][y + current_block.block_position[1]] = e

        return current_game_state

    def check_collision(self, block, next_block_position):
        " Check current block collsion to grid or other blocks"
        current = np.count_nonzero(self.set_current_block_to_gamestate(block))
        next_jump = np.count_nonzero(self.set_current_block_to_gamestate(next_block_position))
        if current > next_jump:
            return True
        return False

    def check_fail(self):
        """
        Check if game is ended.
        """
        if len(set(self.game_state[0])) > 1 or len(set(self.game_state[1])) > 1 or len(set(self.game_state[2])) > 1:
            return True
        else:
            return False
    
    def check_tetris(self, score):
        """
        Check and remove rows with tetris.
        Also add 10p per tetris
        """
        new_game_state = []
        for row in self.game_state:
            if 0 in row:
                new_game_state.append(row)
            else:
                new_game_state.insert(0, [0] * 10)
                score += 1
        self.game_state = new_game_state
        return score


    def check_holes(self, score):
        """
        Check holes in matrix and remove some fitness if found
        """
        def dfs(grid, i, j, h, w):   
            grid[i][j] = 1
            if i + 1 < h and grid[i+1][j] == 0:
                dfs(grid, i+1, j, h, w)
            if i - 1 >= 0 and grid[i-1][j] == 0: 
                dfs(grid, i-1, j, h, w)
            if j + 1 < w and grid[i][j+1] == 0:
                dfs(grid, i, j+1, h, w)
            if j - 1 >= 0 and grid[i][j-1] == 0:    
                dfs(grid, i, j-1, h, w)

        num = 0    
        current_game_state = copy.deepcopy(self.game_state)
        if current_game_state:
            h = len(current_game_state)
            w = len(current_game_state[0])
            for i in range(h):
                for j in range(w):
                    if current_game_state[i][j] == 0:
                        num += 1                    
                        dfs(current_game_state, i, j, h, w)
        return score + - (num * 0.03)

    def check_height(self, score):
        over_height = 0
        for x in self.game_state[0:14]:
            if len(set(x)) > 1:
                over_height += 1
                continue
        
        score = score - (0.01 * over_height)

