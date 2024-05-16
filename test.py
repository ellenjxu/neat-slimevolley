# Test suite for train
import train
import jax
import jax.numpy as jnp
import neat
import numpy as np

key = jax.random.PRNGKey(0)
# create two random nets for testing
config_file = "config-feedforward"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

genome1 = neat.DefaultGenome(1)
genome2 = neat.DefaultGenome(2)
genome1.configure_new(config.genome_config)
genome2.configure_new(config.genome_config)

net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
    
#-----------------------------
    
def single_step():
    """ run one step, render and show picture, and print out observations """
    env = train.SlimeVolley(test=True, max_steps=1, multiagent=True)
    global key
    key, subkey = jax.random.split(key) # update key for each match
    task_state = env.reset(subkey[None,:])
    img = env.render(task_state)
    img.show()

    obs_left, obs_right = task_state.obs
    obs_left = obs_left[0] # for printing
    obs_right = obs_right[0]

    print("Observations for the left agent:")
    print(f"x: {obs_left[0]}, y: {obs_left[1]}, vx: {obs_left[2]}, vy: {obs_left[3]},")
    print(f"bx: {obs_left[4]}, by: {obs_left[5]}, bvx: {obs_left[6]}, bvy: {obs_left[7]},")
    print(f"ox: {obs_left[8]}, oy: {obs_left[9]}, ovx: {obs_left[10]}, ovy: {obs_left[11]}")
    
    print("Observations for the right agent:")
    print(f"x: {obs_right[0]}, y: {obs_right[1]}, vx: {obs_right[2]}, vy: {obs_right[3]},")
    print(f"bx: {obs_right[4]}, by: {obs_right[5]}, bvx: {obs_right[6]}, bvy: {obs_right[7]},")
    print(f"ox: {obs_right[8]}, oy: {obs_right[9]}, ovx: {obs_right[10]}, ovy: {obs_right[11]}")

def single_match():
    """ run a single match of slimevolley and show scores and gif """
    env = train.SlimeVolley(test=True, max_steps=1, multiagent=True) 
    task_reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    scores = [0, 0]

    # modified function below
    # scores, screens = train.play_match(net1, net2, scores, 0, 1, save_gif=True)

    global key
    key, subkey = jax.random.split(key) # update key for each match
    task_state = task_reset_fn(subkey[None,:])
    screens = []

    while True:
        obs_left, obs_right = task_state.obs
        obs_left = obs_left.flatten()[:8] # x,y,vx,vy,bx,by,bvx,bvy
        obs_right = obs_right.flatten()[:8]
        print("Obs left: ", obs_left, "\n")
        print("Obs right: ", obs_right, "\n")
        
        # TODO: features for relative positions
        # ex. rx = bx - x, is ball in front or behind the agent?
        # make sure it is invariant to whether player is on right or left; policy is unchanged
        features_left = train.process_features(obs_left, -1)
        features_right = train.process_features(obs_right, 1)
        print("Features left: ", features_left, "\n")
        print("Features right: ", features_right, "\n")

        action_left = jnp.array(net1.activate(features_left)) # NEAT-python expects flattened
        action_right = jnp.array(net2.activate(features_right))
        # print(action_left, action_right)
        actions = jnp.stack((action_left, action_right), axis=0)
        actions = actions.reshape(1, 2, -1)

        task_state, reward, done = step_fn(task_state, actions)
        
        # print decoded action; game.agent_left.p, saved in Agent.set_action()
        left_agent_state = task_state.game_state.agent_left
        right_agent_state = task_state.game_state.agent_right
        # print(left_agent_state.direction, right_agent_state.direction) # used for inverting the values if left (-1) or right (1)
        
        # print(left_agent_state.vx, left_agent_state.vy)
        # print(right_agent_state.vx, right_agent_state.vy)
        # print(left_agent_state.desired_vx, left_agent_state.desired_vy)
        # print(right_agent_state.desired_vx, right_agent_state.desired_vy)
         
        # update scores with reward; from perspective of agent on the right
        scores[0] -= reward[0].item()
        scores[1] += reward[0].item()
        
        img = env.render(task_state)
        img.show()
        screens.append(img)
            
        if done:
            break
    
    screens[0].save("test.gif", save_all=True, append_images=screens[1:], duration=40, loop=0)

    print(f"Final scores after the match: {scores}")

if __name__ == "__main__":
     
    # single_step()
    single_match()