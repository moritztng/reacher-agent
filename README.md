# Reacher Agent
![Reacher](reacher.gif)

## The Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Getting Started
To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/moritztng/reacher-agent.git
cd reacher-agent/python
pip install .
```

3. Download the Unity Environment
Download the Unity Environment from one of the links below. You need only select the environment that matches your operating system:
- Linux: [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows(32 Bit):[W32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows(64 Bit): [W64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Then, place the file in the GitHub repository, and unzip (or decompress) the file.

## Training
I framed this problem as a Markov Decision Process. Therefore it makes sense to tackle the problem with reinforcement learning. However, in this environment we have a **continuous action space**. Hence, we can't make use of traditional value based methods, like **(Deep-)Q-Learning**. That means we have to use some kind of policy based method. However, in most problems value based methods perform way better than policy based methods. Thus we choose something in between and make use of an **actor critic method**, namely **DDPG**. For more information take a look at the [Notebook](train.ipynb) or the [report](report.md). 

## Testing
You are able to test the agent with the trained weigths by executing `test.py`. 
You can change the number of **episodes** with the following command:
```bash
python test.py --episodes 5
```

## Acknowledgement
This project was part of the Udacity Reinforcement Learning Nanodegree. Without their resources the project would not exist.
