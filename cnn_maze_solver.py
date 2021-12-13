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
from agent_package import agent as a
from CNN10 import CNN10 as CN
import gym
import torch
import numpy as np

env = gym.make("maze-v0")


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
    print("Rewards mean:", np.mean(trajectory_rewards))
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
    exploration_period = 200  # Number of episodes in exploration period
    agent = a.Agent(
        maze_env.original_maze,
        exploration_period,
        starting_epsilon=1,
        buffer_size=buffer_size,
    )  # Build agent
    last_episode = []  # List of actions for replay
    episode_number = 0

    while episode_number < number_of_episodes:
        print(episode_number)
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
                print("Episode: ", episode_number)
                print("Epsilon:", agent.epsilon)
                print("Steps: ", maze_env.step_count)
                episode_steps.append(maze_env.step_count)
                # last_episode = agent.replay_buffer[-steps:,1] #list of actions in ep
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
                    print(maze_env.step_count)
                    agent.update_elite_buffer(maze_env.step_count)
                    print(agent.elite_experience_buffer.size)
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
        if (
            agent.elite_experience_buffer.is_full()
            and exploration_over
            and (episode_number % 10 == 0)
        ):
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
    return (episode_steps, episode_av_reward)


def maze_solver():
    """
    Main function of project, pulls together all packages and executes training
    Returns
    -------
    steps : list
        List of episode seps, used to analyse training success/failure
        Decreasing trend implies learning
    """

    maze_env = gym.make("maze-v0")
    learning_rate = 1e-4  # Hyperparameter for optimiser
    model = CN.ECNN10(1, 4).float()
    ADAM = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = train(maze_env,
                  model,
                  number_of_episodes=2000,
                  discount_factor=0.95,
                  optimiser=ADAM)
    # print(final_episode)
    # agent.replay(final_episode)
    # animate the actions of the agent in final episode
    return steps


episode_steps = []
episode_av_reward = []
steps, rewards = maze_solver()
