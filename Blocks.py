import random
from Block import Block

class Blocks():
    def __init__(self):
        self.blocks = {"O":[[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
                        "Z":[[0,0,0,0],[0,1,1,0],[0,0,1,1],[0,0,0,0]],
                        "S":[[0,0,0,0],[0,0,1,1],[0,1,1,0],[0,0,0,0]],
                        "I":[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
                        "T":[[0,0,0,0],[0,0,1,0],[0,1,1,1],[0,0,0,0]],
                        "J":[[0,0,0,0],[0,1,0,0],[0,1,1,1],[0,0,0,0]],
                        "L":[[0,0,0,0],[0,0,0,1],[0,1,1,1],[0,0,0,0]]}
    
    def get_random_block(self):
        """
        Returns:
            Random block from the list 
        """
        key = random.choice(list(self.blocks.keys()))
        block = Block(self.blocks[key], key)
        return block