import jax
import jax.numpy as jnp
import numpy as np
import neat
import visualize
import sys
import os
import pickle

from datetime import datetime
from slimevolley import SlimeVolley as MultiAgentSlimeVolley
# from evojax.task.ma_slimevolley import MultiAgentSlimeVolley
from evojax import util

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

max_steps = 1000 # 3000
n_rounds = 50 # # of tournament rounds to play; proportional to population size
n_generations = 500
save_gif = False
save_freq = 50

hit_reward = 0.1 # small reward for hitting ball
hit_reward_decay = 0.9 # anneal hit reward over generations

# training and test tasks
train_task = MultiAgentSlimeVolley(test=False, max_steps=max_steps, multiagent=True)
test_task = MultiAgentSlimeVolley(test=True, max_steps=max_steps, multiagent=False)

# initialize random key
key = jax.random.PRNGKey(0)
task_reset_fn = jax.jit(train_task.reset)
step_fn = jax.jit(train_task.step)

#--------------------------------------

def process_features(obs, direction):
    """
    Feature engineering for agent (invariant to the side of the field the agent is on).
    
    obs: Observation from the environment.
    direction: Direction of the agent (-1 for left side, 1 for right side).
    Returns vector of features.
    """

    x,y,vx,vy,bx,by,bvx,bvy,ox,oy,ovx,ovy = obs
    
    # Calculate relative positions and velocities from the agent's perspective
    rel_ball_x = (bx - x) * direction
    rel_ball_y = by - y
    rel_ball_vx = bvx * direction
    rel_ball_vy = bvy
    return np.array([x,y,rel_ball_x, rel_ball_y, rel_ball_vx, rel_ball_vy]) # using 6 features now
    
def play_match(net1, net2, scores, i, j, save_gif=False):
    """
    Play a match of SlimeVolley between two neural networks using the SlimeVolley class.
    Returns reward (indicating winner).
    """
     
    global key
    key, subkey = jax.random.split(key) # update key for each match
    task_state = task_reset_fn(subkey[None,:])
    screens = []
    
    # logger.info(f"Playing match {i} vs {j}")

    while True:
        obs_left, obs_right = task_state.obs
        
        # feature engineering
        obs_left = obs_left.flatten()
        obs_right = obs_right.flatten()
        features_left = process_features(obs_left, -1)
        features_right = process_features(obs_right, 1)

        action_left = jnp.array(net1.activate(features_left)) # NEAT-python expects flattened
        action_right = jnp.array(net2.activate(features_right))

        actions = jnp.stack((action_left, action_right), axis=0)
        actions = actions.reshape(1, 2, -1)

        task_state, reward_win, hits_left, hits_right, done = step_fn(task_state, actions)
        # update scores with reward; from perspective of agent on the right
        # decay hit reward over time
        global hit_reward
        scores[i] = scores[i] - reward_win[0].item() + hits_left[0].item() * hit_reward
        scores[j] = scores[j] + reward_win[0].item() + hits_right[0].item() * hit_reward
        
        if save_gif:
            img = MultiAgentSlimeVolley.render(task_state)
            screens.append(img)
            
        if done:
            break

    # logger.info(f"Scores: {scores}")

    return scores, screens

generation = 0
def fitness(genomes, config):
    """
    Gets the fitness of the genomes by playing a tournament of self-play matches.
    
    genomes: list of (id, genome)
    config: config object
    """

    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes] # create nets from genome representation
    scores = [0] * len(genomes)
    games_played = [0] * len(genomes)

    global generation
    global hit_reward

    # play tournament 
    for _ in range(n_rounds):
        i, j = np.random.choice(len(genomes), 2, replace=False)  
        scores, screens = play_match(nets[i], nets[j], scores, i, j, save_gif=save_gif) 
        games_played[i] += 1
        games_played[j] += 1

        if save_gif:
            gif_filename = f"generation_{generation}_match_{i}_vs_{j}.gif"
            gif_file = os.path.join(log_dir, gif_filename)
            screens[0].save(gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0)
 
    # update genome fitness from scores
    for i, score in enumerate(scores):
        if games_played[i] > 0:
            genomes[i][1].fitness = score / games_played[i] # normalize; win rate
        else:
            genomes[i][1].fitness = 0

    # every 10th generation, save best net
    if generation % save_freq == 0:
        logger.info(f"Generation {generation}")
        logger.info(f"Scores: {scores}")

        best_idx = np.argmax(scores)
        best_net = neat.nn.FeedForwardNetwork.create(genomes[best_idx][1], config)
        with open(os.path.join(log_dir, f"best_{generation}.pkl"), "wb") as f:
            pickle.dump(genomes[best_idx][1], f)

        plot_graph(genomes[best_idx][1], None)

        if generation-save_freq >= 0:
            # load and play against previous best
            with open(os.path.join(log_dir, f"best_{generation-save_freq}.pkl"), "rb") as f:
                prev_best_genome = pickle.load(f)

            prev_net = neat.nn.FeedForwardNetwork.create(prev_best_genome, config)
            gen_scores = [0,0]
            winner, screens = play_match(prev_net, best_net, gen_scores, 0, 1, save_gif=True)
            gif_filename = f"generation_{generation-save_freq}_vs_{generation}_best.gif"
            gif_file = os.path.join(log_dir, gif_filename)
            screens[0].save(gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0)
            logger.info(f"Winner: {winner}")

    hit_reward = hit_reward * hit_reward_decay
    generation += 1

def play_test(best_net, gif_file=None):
    """
    Play a match of SlimeVolley between the best neural network and in-built policy.
    Returns reward (indicating winner).
    """
    logger.info("Playing test match...") 

    task_reset_fn = jax.jit(test_task.reset)
    step_fn = jax.jit(test_task.step)
    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    screens = []
    for _ in range(1, max_steps):
        obs = task_state.obs.flatten()
        processed_obs = process_features(obs, 1)
        action = jnp.array(best_net.activate(processed_obs))
        # reshape action
        action = action.reshape(1,-1)
        task_state, reward, done = step_fn(task_state, action)
        screens.append(MultiAgentSlimeVolley.render(task_state))

    logger.info(f"Reward: {reward}")

    if gif_file is not None:
        screens[0].save(gif_file, save_all=True, append_images=screens[1:],
                    duration=40, loop=0)
    
    return reward
    
def plot_graph(genome, stats):
    node_names = {0: "forward", 1: "backward", 2: "jump", -1: "x", -2: "y", -3: "vx", -4: "vy", -5: "bx", -6: "by", -7: "bvx", -8: "bvy", -9: "ox", -10: "oy", -11: "ovx", -12: "ovy"}
    # node_names = {0: "forward", 1: "backward", 2: "jump", -1: "rx", -2: "ry", -3: "rvx", -4: "rvy"}
    plot_name = f"generation_{generation}"
    visualize.draw_net(config, genome, view=True, node_names=node_names, filename=os.path.join(log_dir, plot_name))

    if stats is not None:
        # save stats to pickle
        with open(os.path.join(log_dir, f"stats_{generation}.pkl"), "wb") as f:
            pickle.dump(stats, f)
        visualize.plot_stats(stats, ylog=True, view=True)
        visualize.plot_species(stats, view=True)

def run():
    """
    Run NEAT with max generations
    """
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(save_freq, filename_prefix=os.path.join(log_dir,"checkpoint")))
    
    winner = p.run(fitness, n_generations)

    with open(os.path.join(log_dir, "best.pkl"), "wb") as f:
        pickle.dump(winner, f)
    logger.info(f"Winner: {winner}")

    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    play_test(best_net)
    
    plot_graph(winner, stats) 

if __name__ == "__main__":
    config_file = "config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
 
    log_dir = "../log/neat" + str(datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = util.create_logger(name='NEATSlimeVolley', log_dir=log_dir)
    logger.info("Logging to {}".format(log_dir))
    logger.info("number of inputs: {}, max steps: {}, n_rounds: {}, n_generations: {}, population size: {}".format(len(config.genome_config.input_keys), max_steps, n_rounds, n_generations, config.pop_size))
    
    run() 
