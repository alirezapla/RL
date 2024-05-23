import numpy as np
import random
import matplotlib.pyplot as plt
import collections


GAMMA = 0.92


class Algorithm:
    def __init__(self, alpha, epsilon) -> None:
        self.total_reward = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.successful_steps = []

    def _set_total_reward(self, reward):
        self.total_reward += reward

    def reset_total_reward(self):
        self.total_reward = 0

    def policy_evaluation(self, done, reward, step, state, q_states, q_state):
        pass

    def _decay_epsilon(self, curr_eps) -> float:
        if curr_eps < 1e-2:
            return curr_eps
        return curr_eps * 0.89

    def _greedy(self, state, q_states) -> tuple:
        q_vals = [q_states[state + (action,)] for action in range(4)]
        return max(q_vals)

    def __repr__(self) -> str:
        pass


class QLearning(Algorithm):
    def __init__(self, alpha, epsilon) -> None:
        super().__init__(alpha, epsilon)

    def policy_evaluation(self, done, reward, step, state, q_state, q_states):
        if done:
            if self.total_reward > 0:
                print(reward, step, self.total_reward)
                self.successful_steps.append(step)
            self.epsilon = self._decay_epsilon(self.epsilon)
            q_states[q_state] += self.alpha * (reward - q_states[q_state])
        else:
            q_states[q_state] += self.alpha * (
                reward + GAMMA * self._greedy(state, q_states) - q_states[q_state]
            )
        self._set_total_reward(reward)
        return q_states

    def __repr__(self) -> str:
        return "Q-Learning"


class SARSA(Algorithm):
    def __init__(self, alpha, epsilon) -> None:
        super().__init__(alpha, epsilon)

    def policy_evaluation(self, done, reward, step, state, q_state, q_states):
        if done:
            if self.total_reward > 0:
                print(reward, step, self.total_reward)
                self.successful_steps.append(step)
            self.epsilon = self._decay_epsilon(self.epsilon)
            q_states[q_state] += self.alpha * (reward - q_states[q_state])
        else:
            q_states[q_state] += self.alpha * (
                reward + GAMMA * self._greedy(state, q_states) - q_states[q_state]
            )
        self._set_total_reward(reward)
        return q_states

    def __repr__(self) -> str:
        return "SARSA"


class Agent:
    def __init__(self, env, alg: Algorithm) -> None:
        self.env = env
        self.q_states = collections.defaultdict(float)
        self.last_state = None
        self.alg = alg
        self.observation = [0, 0, 0, 0, 0, 0, 0, 0]

    def run_alg(self, step: int) -> tuple:
        self.last_state = self._discretize(self.observation)
        action = self._action(self.last_state)
        q_state = self.last_state + (action,)
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        state = self._discretize(self.observation)
        done = terminated or truncated
        self.q_states = self.alg.policy_evaluation(
            done, reward, step, state, q_state, self.q_states
        )
        return done

    def plot(self):
        if len(self.alg.successful_steps) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(
                self.alg.successful_steps,
                [n for n in range(len(self.alg.successful_steps))],
                label=f"{self.alg} α={self.alg.alpha}",
            )

        plt.xlabel("Time Steps")
        plt.ylabel("Number of Successful Steps")
        plt.legend()
        plt.title("Successful Steps over Time Steps")
        plt.show()

    def _action(self, state) -> int:
        prob = np.random.random()
        if prob < self.alg.epsilon:
            return random.choice(range(self.env.action_space.n))
        else:
            q_vals = [
                self.q_states[state + (action,)]
                for action in range(self.env.action_space.n)
            ]
            return np.argmax(q_vals)

    def test_policy(self, iteration: int) -> list:
        total_reward = []
        s = 0
        state = self._discretize(self.env.reset()[0])
        for _ in range(iteration):
            action = self._action(state)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                s += 1 if reward != -100 else 0
                state = self._discretize(self.env.reset()[0])
            else:
                state = self._discretize(observation)
            total_reward.append(round(reward, 5))
        print(total_reward)
        print(s)
        return total_reward

    def _discretize(self, state) -> tuple:
        return (
            min(3, max(-2, int(state[0] / 0.15))),
            min(3, max(-2, int(state[1] / 0.15))),
            min(3, max(-2, int(state[2] / 0.15))),
            min(3, max(-2, int(state[3] / 0.15))),
            min(3, max(-2, int(state[4] / 0.15))),
            min(3, max(-2, int(state[5] / 0.15))),
            int(state[6]),
            int(state[7]),
        )
