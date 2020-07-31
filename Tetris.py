import pygame
import os
import neat
from GameState import GameState
from Blocks import Blocks

WIN_WIDTH = 700
WIN_HEIGHT = 645

key_timeout = {}
def getPressed(keys, key, timeout):
    global key_timeout

    if keys[key] == False:
        return False

    current_time = pygame.time.get_ticks()

    if key in key_timeout and key_timeout[key] > current_time:
        return False

    key_timeout[key] = current_time + timeout
    return True

def main(genomes, config):
    blocks = Blocks()
    nets = []
    ge = []
    games = []
    current_blocks = []
    next_blocks = []
    running = True
    clock = pygame.time.Clock()
    clock.tick(99999)  

    for index, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)
        games.append(GameState())
        current_blocks.append(blocks.get_random_block())
        next_blocks.append(blocks.get_random_block())

    while running:
        win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
        for i, game_state in enumerate(games):
            running = True

            # Next Block
            if not next_blocks[i]:
                next_blocks[i] = blocks.get_random_block()
            
            input_values = []
            for v in game_state.game_state[14:]:
                input_values += v

            if not current_blocks[i].dropped:
                for v in current_blocks[i].block_matrix:
                    input_values += v

                for v in next_blocks[i].block_matrix:
                    input_values += v
                
                for v in current_blocks[i].block_position:
                    input_values.append(v)

                output = nets[i].activate(input_values)
                max_value_rotate = max(output[0:4])

                if output[0] ==  max_value_rotate:
                    current_blocks[i].rotate_block(game_state)
                    current_blocks[i].rotate_block(game_state)
                    current_blocks[i].rotate_block(game_state)
                    current_blocks[i].rotate_block(game_state)
                elif output[1] ==  max_value_rotate:
                    current_blocks[i].rotate_block(game_state)
                    current_blocks[i].rotate_block(game_state)
                    current_blocks[i].rotate_block(game_state)
                elif output[2] ==  max_value_rotate:
                    current_blocks[i].rotate_block(game_state)
                    current_blocks[i].rotate_block(game_state)
                elif output[3] ==  max_value_rotate:
                    current_blocks[i].rotate_block(game_state)

                current_blocks[i].calculate_width()
                
                left_index = 9 - current_blocks[i].calculate_width()

                max_left_move = max(output[4:left_index])
                
                right_index = 15 - current_blocks[i].calculate_width()

                max_right_move = max(output[10:right_index])

                max_value_move = max([max_right_move,left_index,output[4]])

                if output[9] ==  max_value_move:
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                elif output[8] ==  max_value_move:
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                elif output[7] ==  max_value_move:
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                elif output[6] ==  max_value_move:
                    current_blocks[i].move_left(game_state)
                    current_blocks[i].move_left(game_state)
                elif output[5] ==  max_value_move:
                    current_blocks[i].move_left(game_state)
                elif output[4] ==  max_value_move:
                    pass
                elif output[14] ==  max_value_move:
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                elif output[13] ==  max_value_move:
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                elif output[12] ==  max_value_move:
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                elif output[11] ==  max_value_move:
                    current_blocks[i].move_right(game_state)
                    current_blocks[i].move_right(game_state)
                elif output[10] ==  max_value_move:
                    current_blocks[i].move_right(game_state)

                current_blocks[i].dropped = True

            if i == 0:
                game_state.draw_window(win, current_blocks[i], ge[i].fitness, next_blocks[i])

            # Move Block
            if not current_blocks[i].move_down(game_state):
                game_state.game_state = game_state.set_current_block_to_gamestate(current_blocks[i])
                current_blocks[i] = next_blocks[i]
                next_blocks[i] = None
                game_state.check_tetris()
                game_state.check_holes()
                game_state.check_height()
                game_state.check_pikes()

                a = -0.510066
                b = 0.760666
                c = -0.35663
                d = -0.184483
                ge[i].fitness = (a * game_state.field_used) + (b * game_state.tetris) + (c * game_state.holes) + (c * game_state.difference)

            if game_state.check_fail():
                ge[i].fitness -= 1
                nets.pop(i)
                ge.pop(i)
                games.pop(i)
                if len(games) == 0:
                    running = False
                    pygame.display.quit()
                    break


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,500)
    print("Winner Winner!")
    with open("genomes.txt", "w") as file1: 
        # Writing data to a file 
        file1.write(str(winner)) 

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)