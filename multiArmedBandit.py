import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
import time

start_time = time.time()


class Bandit:
    '''This class is an implementation of a k-armed bandit.
    '''

    def __init__(self, number_of_arms: int = 1, location: float = 0.0, scale: float = 1.0) -> None:
        self.number_of_arms = number_of_arms
        self.means = []
        self.means = random.normal(location, scale, number_of_arms).tolist()

    def __repr__(self) -> str:
        return f'bandit with {self.number_of_arms} arms and means {self.means}'

    def pull_arm(self, arm: int) -> float:
        return random.normal(self.means[arm], 1.0)

    def action_value(self, arm: int) -> float:
        return self.means[arm]

    def maximum_value(self) -> float:
        return max(self.means)

    def arm_with_maximum_value(self) -> int:
        return self.means.index(max(self.means))


def find_arm_with_max_estimated_value(rewards, frequencies) -> int:
    max_estimated_value = -1000000
    max_index = -1
    for i in range(len(rewards)):
        if frequencies[i] > 0 and rewards[i] / frequencies[i] > max_estimated_value:
            max_estimated_value = rewards[i] / frequencies[i]
            max_index = i
    return max_index


# main code
# initialize the bandit
# loop over N steps
#   choose lever to pull
#   pull lever
#   update estimated value for lever

number_of_runs = 2000
number_of_steps = 1000
arms: int = 10
# random_selection_prob = 0.1
random_selection_prob = 0.01
average_rewards = [0]*number_of_steps

for run in range(number_of_runs):
    bandit = Bandit(number_of_arms=arms)
    # print(bandit)
    # print(f'arm with max value is {bandit.arm_with_maximum_value()} with '
    #       f'value {bandit.action_value(bandit.arm_with_maximum_value())}')

    sum_of_rewards = [0] * arms
    frequency = [0] * arms
    rewards = []
    for step in range(number_of_steps):
        # print(f'arm with max est value is {find_arm_with_max_estimated_value(sum_of_rewards, frequency)}')
        if random.random() < random_selection_prob:
            # choose arm randomly
            chosen_arm = random.randint(0, arms)
        else:
            # choose arm with largest estimated value
            chosen_arm = find_arm_with_max_estimated_value(sum_of_rewards, frequency)

        reward = bandit.pull_arm(chosen_arm)
        # print(f'Reward for chosen arm {chosen_arm} is {reward}')
        sum_of_rewards[chosen_arm] += reward
        frequency[chosen_arm] += 1
        # print(sum_of_rewards)
        # print(frequency)
        rewards.append(reward)
    for i in range(number_of_steps):
        average_rewards[i] += rewards[i]

average_rewards[:] = [x / number_of_runs for x in average_rewards]
print(f'Time to execute data processing: {time.time() - start_time} seconds')

# print(f'Estimated values after {number_of_steps} steps:')
# print(sum_of_rewards)
# print(frequency)
# for i in range(arms):
#     if frequency[i] > 0:
#         print(f'{i}: {frequency[i]} pulls. {sum_of_rewards[i] / frequency[i]} ')

# print(rewards)

plt.style.use('seaborn-whitegrid')

df = pd.DataFrame(list(zip(range(number_of_steps), average_rewards)),
                  columns=['step', 'reward'])
# print(df.head())
plt.plot('step', 'reward', data=df,  linestyle='-', marker="")
plt.xlabel('Step')
plt.title('Average reward')
plt.grid(True)
plt.show()
