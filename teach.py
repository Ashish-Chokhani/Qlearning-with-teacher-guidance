#!/usr/bin/env python3
"""
teach.py
"""

import numpy as np
from env import StaticGridEnv
from utils import plot_rewards, plot_heatmap


np.random.seed(42)


# ---------------------- ENVIRONMENT SETUP ----------------------
def init_environment(seed=42):
    """Set up the environment. We'll fix the seed to keep results consistent."""
    np.random.seed(seed)
    env = StaticGridEnv(seed)
    return env

# ---------------------- ACTION SELECTION ----------------------
def epsilon_greedy_action(q_table, state, epsilon, action_space):
    """
    Choose an action based on the epsilon-greedy strategy:
    - With probability epsilon, we explore (pick a random action).
    - Otherwise, we exploit (choose the best-known action).
    """
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    x, y = state
    return np.argmax(q_table[x, y])


def teacher_action(teacher_q_table, state, availability, accuracy, action_space):
    """
    Simulate a teacher giving advice:
    - Based on availability, we may or may not use the teacher.
    - The teacher gives the correct action with a certain accuracy.
    """
    x, y = state
    use_teacher = np.random.rand() < availability if availability > 0 else False
    if not use_teacher:
        return None, False
    correct_action = np.argmax(teacher_q_table[x, y])
    if np.random.rand() < accuracy:
        return correct_action, True
    other_actions = [a for a in range(action_space) if a != correct_action]
    return np.random.choice(other_actions), True


# ---------------------- Q-TABLE UPDATE ----------------------
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Classic Q-learning update rule:
    We adjust our estimate toward the observed reward + estimated future value.
    """
    x, y = state
    nx, ny = next_state
    best_next_action = np.max(q_table[nx, ny])
    q_table[x, y, action] += alpha * (reward + gamma * best_next_action - q_table[x, y, action])


# ---------------------- TRAINING LOOP ----------------------
def run_training_episode(env, q_table, max_steps, epsilon, alpha, gamma, action_selector):
    """
    Run one episode of training using vanilla Q-learning.
    We select actions with the epsilon-greedy policy and update our Q-table.
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = action_selector(q_table, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
        state = next_state
        total_reward += reward
        steps += 1

    return total_reward, done, steps


def run_training_episode_with_teacher(env, q_table, teacher_q_table, max_steps, epsilon,
                                      alpha, gamma, availability, accuracy):
    """
    Similar to standard training, but we check if the teacher is available
    to give advice before selecting our action.
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        teacher_act, used_teacher = teacher_action(teacher_q_table, state, availability, accuracy, env.action_space)
        if used_teacher:
            action = teacher_act
        else:
            action = epsilon_greedy_action(q_table, state, epsilon, env.action_space)

        next_state, reward, done, _ = env.step(action)
        update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
        state = next_state
        total_reward += reward
        steps += 1

    return total_reward, done, steps

############################################################################
######                Train agent using Q-learning                    ######
############################################################################

def train_agent(max_total_steps, initial_epsilon=1.0, epsilon_decay_rate=0.995):
    """
    Main Q-learning training loop without teacher guidance.
    We decay epsilon over time to shift from exploration to exploitation.
    """
    env = init_environment()
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space))

    alpha = 0.3
    gamma = 0.99
    epsilon = initial_epsilon
    epsilon_min = 0.05
    max_steps_per_episode = 100

    reward_per_episode = []
    total_steps = 0
    total_steps_all = 0
    successful_episodes = 0
    total_episodes = 0

    while total_steps < max_total_steps:
        reward, done, steps = run_training_episode(
            env, q_table, max_steps_per_episode, epsilon, alpha, gamma,
            lambda q, s, eps: epsilon_greedy_action(q, s, eps, env.action_space)
        )

        reward_per_episode.append(reward)
        if done:
            successful_episodes += 1
        total_episodes += 1
        total_steps += steps
        total_steps_all += steps
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)

    success_rate = successful_episodes / total_episodes
    avg_steps_per_episode = total_steps_all / total_episodes
    return q_table, reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######               Evaluate agent after training                    ######
############################################################################
def evaluate_agent(q_table, max_total_steps, render=False, availibility= None, accuracy= None):
    """
    Evaluate a trained agent over multiple episodes.
    If render=True, we'll periodically show the environment.
    """
    env = init_environment()
    total_steps = 0
    max_steps_per_episode = 100

    total_reward = 0
    successful_episodes = 0
    total_episodes = 0
    total_steps_all = 0

    while total_steps < max_total_steps:
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps_per_episode:
            x, y = state
            action = np.argmax(q_table[x, y])
            state, reward, done, _ = env.step(action)
            
            if render and total_episodes % 50 == 0:
                env.render(
                    episode=total_episodes,
                    learning_type="Q-learning",
                    availability=availibility,
                    accuracy=accuracy
                )
            
            episode_reward += reward
            steps += 1
            total_steps += 1
            total_steps_all += 1

        total_reward += episode_reward
        total_episodes += 1
        if done:
            successful_episodes += 1

    avg_reward_per_episode = total_reward / total_episodes
    success_rate = (successful_episodes / total_episodes) * 100
    avg_steps_per_episode = total_steps_all / total_episodes
    return avg_reward_per_episode, success_rate, avg_steps_per_episode



############################################################################
######        Train agent using Q-learning with Teacher Advice        ######
############################################################################

def train_agent_with_teacher(teacher_q_table, max_total_steps, availability, accuracy,
                             initial_epsilon=1.0, epsilon_decay_rate=0.995):
    """
    Train the agent using a teacher that sometimes gives advice.
    This helps us see how useful a teacher is at different levels of accuracy.
    """
    env = init_environment()
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space))

    alpha = 0.3
    gamma = 0.99
    epsilon = initial_epsilon
    epsilon_min = 0.05
    max_steps_per_episode = 100

    reward_per_episode = []
    total_steps = 0
    total_steps_all = 0
    successful_episodes = 0
    total_episodes = 0

    while total_steps < max_total_steps:
        reward, done, steps = run_training_episode_with_teacher(
            env, q_table, teacher_q_table, max_steps_per_episode, epsilon, alpha,
            gamma, availability, accuracy
        )

        reward_per_episode.append(reward)
        if done:
            successful_episodes += 1
        total_episodes += 1
        total_steps += steps
        total_steps_all += steps
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)

    avg_reward_per_episode = np.mean(reward_per_episode)
    success_rate = successful_episodes / total_episodes
    avg_steps_per_episode = total_steps_all / total_episodes
    return q_table, avg_reward_per_episode, success_rate, avg_steps_per_episode

############################################################################
######       Evaluate agent with varying teacher's availability and accuracy        ######
############################################################################

def evaluate_agent_with_teacher(teacher_q_table, TEACHER_AVAILABILITY, TEACHER_ACCURACY, initial_epsilon=1.0, epsilon_decay_rate=0.995):
    """
    Run a series of experiments to see how varying teacher availability and accuracy
    affects training and evaluation performance.
    """
    avg_reward_train = np.zeros((len(TEACHER_AVAILABILITY), len(TEACHER_ACCURACY)))
    avg_reward_eval = np.zeros((len(TEACHER_AVAILABILITY), len(TEACHER_ACCURACY)))
    records = []

    for i, avail in enumerate(TEACHER_AVAILABILITY):
        for j, acc in enumerate(TEACHER_ACCURACY):
            agent_q_table, train_avg_reward, _, _ = train_agent_with_teacher(
                teacher_q_table=teacher_q_table,
                max_total_steps=11900,
                availability=avail,
                accuracy=acc,
                initial_epsilon=initial_epsilon,
                epsilon_decay_rate=epsilon_decay_rate
            )
            eval_avg_reward, _, _ = evaluate_agent(agent_q_table, max_total_steps=1000, render=False,availibility= avail, accuracy= acc)

            avg_reward_train[i, j] = train_avg_reward
            avg_reward_eval[i, j] = eval_avg_reward

            records.append({
                "Availability": avail,
                "Accuracy": acc,
                "Train Avg Reward": train_avg_reward,
                "Eval Avg Reward": eval_avg_reward
            })
    
    return records, avg_reward_train, avg_reward_eval

############################################################################
######                        Main Function                           ######
############################################################################

def main():
    """
    We train the agent, evaluate its performance,
    and then experiment with different teacher configurations.
    """
    
    TEACHER_AVAILABILITY = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    TEACHER_ACCURACY = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    initial_epsilon = 1.0
    epsilon_decay_rate = 0.995

    q_table, reward_per_episode, success_rate, avg_steps_per_episode = train_agent(
        50000,
        initial_epsilon=initial_epsilon,
        epsilon_decay_rate=epsilon_decay_rate
    )

    plot_rewards(reward_per_episode, "Reward per Episode for Q-learning")

    avg_reward_per_episode, success_rate, avg_steps_per_episode = evaluate_agent(q_table, 10000, render=True)

    print("Average Reward per Episode:", avg_reward_per_episode)
    print("Success Rate:", success_rate)
    print("Steps per Episode:", avg_steps_per_episode)
    
    
    records, avg_reward_train, avg_reward_eval = evaluate_agent_with_teacher(q_table, TEACHER_AVAILABILITY, TEACHER_ACCURACY, initial_epsilon=initial_epsilon, epsilon_decay_rate=epsilon_decay_rate)
    
    plot_heatmap(avg_reward_train,
             x_labels=TEACHER_ACCURACY,
             y_labels=TEACHER_AVAILABILITY,
             title="Training Avg Reward for Different Teacher Availability and Accuracy\n(Q-learning Teacher)",
             xlabel="Accuracy",
             ylabel="Availability",
             )

    plot_heatmap(avg_reward_eval,
             x_labels=TEACHER_ACCURACY,
             y_labels=TEACHER_AVAILABILITY,
             title="Evaluation Avg Reward for Different Teacher Availability and Accuracy\n(Q-learning Teacher)",
             xlabel="Accuracy",
             ylabel="Availability",
             )

    
    # Uncomment this if we want to save the experiment results:
    # df_results = pd.DataFrame(records)
    # df_results.to_csv("teacher_training_results.csv", index=False) ## Comment out this line if required to save df


if __name__ == '__main__':
    main()