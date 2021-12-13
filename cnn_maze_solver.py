"""
I, Elliot Hicks, have read and understood the School's Academic Integrity
Policy, as well as guidance relating to this module, and confirm that this
submission complies with the policy. The content of this file is my own
original work, with any significant material copied or adapted from other
sources clearly indicated and attributed.
Author: Elliot Hicks
Project title: RL_CNN_maze_solver
Date: 13/12/2021
"""
from agent_package import agent as ag
from CNN10 import CNN10 as CN
import gym
from gym_maze_package import gym_maze
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make('maze-v0')


def calculate_values(trajectory_rewards, discount_factor):
    """

    Parameters
    ----------
    trajectory_rewards : NumPy ndarray
        List of rewards for steps along trajectory
    discount_factor : float
        How much rewards in the future are discounted,
        helps identify key steps.

    Returns
    -------
    trajectory_rewards : NumPy ndarray
        Updates rewards as:
            R_{t'} = SUM_{t = t'}^{T}(r_t*discount_factor^{t'-t})

    """
    i = 1
    while i < len(trajectory_rewards):
        # disount the future rewards and add them to current value
        trajectory_rewards[-i - 1] += discount_factor * trajectory_rewards[-i]
        i += 1
    return trajectory_rewards


def loss_fn(model, states, action_labels, values):
    """
    Parameters
    ----------
    model : PyTorch Module object
        CNN designed to take single channel 10x10 images
    states : list of  NumPy ndarrays
        List of states from randomly sampled transitions
    action_labels : int
        encoded labels for actions: 0,1,2,3 = "up", "right", "left", "down"
    values : list of floats
        Discunted rewards along trajectories which have been randomly sampled

    Returns
    -------
    loss : Torch 1D Tensor
        Calculated using 0-bias loss funtion , inspired by REINFORCE paper:
            https://link.springer.com/article/10.1023/A:1022672621406

    """
    loss = 0
    for i in range(len(values)):
        probability_i = model(states[i])
        loss -= torch.mul(torch.log(probability_i[0][action_labels[i]]),
                          values[i])
    return loss


def train(maze_env, model, number_of_episodes, discount_factor, optimiser):
    """
    Parameters
    ----------
    maze_env : OpenAI gym Env object
        Interactive custom Gym environment,(converted maze_maker maze)
    model : PyTorch Module object
        CNN with architecture designed for maze dimensions
    number_of_episodes : int
        (AKA number of trials),hyperparameter,set by user for lengt of episodes
    discount_factor : float
        Discount rate to help identify key steps in trajectories
    optimiser : PyTorch Optimizer object
        Optimizer for model, for this application ADAM or SGD recommended
    Returns
    -------
    episode_steps : list
        list of steps taken in each episode, = max_steps if trajectory times
        out, used to track progress. Decreases suggest improvements.
    episode_av_reward : list
        list of average rewards from trajectories in training (after
        discounting). Used to track progress. Increases suggest improvemnts/

    """
    buffer_size = 3000  # Agent main memory buffer capacity
    exploration_period = 20  # Number of episodes in exploration period
    agent = ag.Agent(
        maze_env.original_maze,
        exploration_period,
        starting_epsilon=1,
        buffer_size=buffer_size,
    )  # Build agent
    best_trajectory = []
    episode_number = 0

    while episode_number < number_of_episodes:
        maze_env.reset()  # Start/Restart maze
        agent.position = np.array([1, 1])  # Set/Reset agent position
        total_transitions = 0
        episode_done = False
        while not episode_done:
            # Map state to action probabilities using policy function:
            action_probabilities = model(maze_env.state).detach().numpy()[0]
            # Set invalid action probs to zero, renomalise:
            action_probabilities = agent.test_actions(action_probabilities)
            # Choose action according to epsilon-greedy:
            action, action_label = agent.choose_action(action_probabilities)
            agent.position += action  # Update agent position
            state, action, reward, episode_done = maze_env.step(action)
            agent.replay_buffer.add([state, action_label, reward, episode_done])
            total_transitions += 1

            if episode_done:
                print(("| Episode Number: {0} | Steps: {1} " +
                       "| Epsilon: {2:.2f} |").format(episode_number,
                                                      maze_env.step_count,
                                                      agent.epsilon))
                episode_steps.append(maze_env.step_count)
                #  Update trajctory values using discounted rewards:
                updated_trajectory_values = calculate_values(
                    agent.replay_buffer[-maze_env.step_count:, 2],
                    discount_factor)
                #  Update rewards column of memory buffer with new values:
                agent.replay_buffer.update_values(
                    maze_env.step_count,
                    updated_trajectory_values)
                #  Track progress by logging trajectory rewards:
                episode_av_reward.append(np.mean(updated_trajectory_values))

                # Test 'elite' trajectories and store their transitions:
                if maze_env.step_count < np.percentile(episode_steps, 10):
                    agent.elite_experience_buffer.add_last_episode(maze_env.step_count)
                if maze_env.step_count == min(episode_steps):
                    best_trajectory = agent.replay_buffer[-maze_env.step_count:, 0]

        #  Update epsilon according to epsilon scheme:
        agent.update_epsilon(episode_number, number_of_episodes)

        exploration_over = episode_number >= exploration_period
        if exploration_over and (episode_number % 5 == 0):
            # Batch train CNN every 5 episodes after exploration period:
            training_batch = agent.replay_buffer.random_batch()
            states = training_batch[:, 0]
            action_labels = training_batch[:, 1]
            values = training_batch[:, 2]
            loss = loss_fn(model, states, action_labels, values)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if (agent.elite_experience_buffer.is_full() and exploration_over and
            (episode_number % 10 == 0)):
            # Train CNN after every 10 episodes:
            training_batch = agent.elite_experience_buffer.random_batch(400)
            states = training_batch[:, 0]
            action_labels = training_batch[:, 1]
            values = training_batch[:, 2]
            loss = loss_fn(model, states, action_labels, values)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        #  Increment episode counter:
        episode_number += 1
    return (episode_steps, episode_av_reward, maze_env.original_maze,
            best_trajectory)


def maze_solver():
    """
    Main function of project, pulls together all packages and executes training
    Returns
    -------
    steps : list
        List of episode seps, used to analyse training success/failure
        Decreasing trend implies learning
    """

    maze_env = gym.make('maze-v0')
    learning_rate = 1e-4  # Hyperparameter for optimiser
    model = CN.ECNN10(1, 4).float()
    ADAM = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = train(maze_env,
                  model,
                  number_of_episodes=300,
                  discount_factor=0.95,
                  optimiser=ADAM)

    return steps


def animate_imshow(frame):
    image.set_array(best_trajectory[frame])
    return [image]


episode_steps = []
episode_av_reward = []
steps, rewards, maze, best_trajectory = maze_solver()
shortest_length = len(best_trajectory)
best_episode = np.argmin(steps)
figure = plt.figure()
plt.axis("off")
plt.title("RL_CNN_maze_solver best solution:" +
          " \nEpisode {0} (steps = {1})".format(best_episode, shortest_length))
image = plt.imshow(best_trajectory[0], interpolation='none')
animated_maze = animation.FuncAnimation(figure, animate_imshow,
                                        len(best_trajectory)-1,
                                        interval=200)

figure.show()

"""
Implementation of a user function where they can load/save mazes with pickle
and possibly open up a file explorer GUI would be the final step in this code.
The code has no user functions, this was not the plan but priority was given to
getting the model working. Unfortunately, I was unable to stabalize training in
time.
"""
