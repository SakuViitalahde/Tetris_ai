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
            
            for v in current_blocks[i].block_matrix:
                input_values += v

            for v in next_blocks[i].block_matrix:
                input_values += v
            
            for v in current_blocks[i].block_position:
                input_values.append(v)

            output = nets[i].activate(input_values)
            max_value = max(output)

            if output[0] ==  max_value and not current_blocks[i].dropped:
                current_blocks[i].move_left(game_state)
            if output[1] ==  max_value and not current_blocks[i].dropped:
                current_blocks[i].move_right(game_state)
            if output[2] ==  max_value and not current_blocks[i].dropped:
                current_blocks[i].dropped = True
            if output[3] ==  max_value and not current_blocks[i].dropped:
                current_blocks[i].rotate_block(game_state)

            if i == 0:
                game_state.draw_window(win, current_blocks[i], ge[i].fitness, next_blocks[i])

            # Move Block
            if not current_blocks[i].move_down(game_state):
                game_state.game_state = game_state.set_current_block_to_gamestate(current_blocks[i])
                current_blocks[i] = next_blocks[i]
                next_blocks[i] = None
                ge[i].fitness += 1
                ge[i].fitness = game_state.check_holes(ge[i].fitness)
                ge[i].fitness = game_state.check_height(ge[i].fitness)
                ge[i].fitness = game_state.check_pikes(ge[i].fitness)

            ge[i].fitness = game_state.check_tetris(ge[i].fitness)

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
    print(winner)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)