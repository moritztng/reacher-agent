from unityagents import UnityEnvironment
import numpy as np
import torch, time, argparse
import torch.nn as nn
from models import Actor

parser = argparse.ArgumentParser(description='Testing The Trained Model In Unity')
parser.add_argument('--episodes', required=True, type=int, help='Number of Episodes the Agent should play')

if  __name__ == "__main__":
    args = vars(parser.parse_args())

    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    actor = Actor(state_size, action_size).to(device)
    actor.eval()
    actor.load_state_dict(torch.load('checkpoint.pth'))

    with torch.no_grad():
        for i in range(args['episodes']):
            env_info = env.reset(train_mode=False)[brain_name]
            state = torch.tensor(env_info.vector_observations[0]).float().to(device)
            score = 0
            while True:
                action = actor(state.unsqueeze(0))
                env_info = env.step(action.to('cpu').numpy())[brain_name]

                reward = env_info.rewards[0]
                score += reward
                done = float(env_info.local_done[0])
                next_state = env_info.vector_observations[0]
                state = torch.tensor(next_state).float().to(device)
                if done:
                    break
            print('Episode: {}, Score: {}'.format(i+1, score))
