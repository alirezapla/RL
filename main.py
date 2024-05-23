import statistics
import gym as gym
from agent import Agent, QLearning, SARSA

EPISODE = 100
DEBUG = False
STEPS = 100_000


def build_game():
    render = None
    if DEBUG:
        render = "human"
    return gym.make("LunarLander-v2", render_mode=render)


def alg_options(env):
    algorithm = int(input("select algorithm\n- 1-> SARSA\n- 2-> Q-learning\n"))
    if algorithm == 1:
        alg = SARSA(0.05, 1)
    elif algorithm == 2:
        alg = QLearning(0.05, 1)
    else:
        raise Exception("illegal option")
    return Agent(env, alg)


def play_game(agent: Agent, env):
    episodes = 0
    for step in range(STEPS):
        done = agent.run_alg(step)
        if done:
            episodes += 1
            print("====", "end of episode", episodes, agent.alg.total_reward, "====")
            agent.alg.reset_total_reward()
            env.reset()
    print(agent.alg.successful_steps)
    # plot_info
    agent.plot()


if __name__ == "__main__":

    env = build_game()
    env.action_space.seed(42)
    print(env.observation_space.low, env.observation_space.high)
    observation, info = env.reset(seed=42)
    agent = alg_options(env)
    observation, info = env.reset()
    play_game(agent, env)
    total_reward = agent.test_policy(1000)
    print(statistics.mean(total_reward), statistics.variance(total_reward))
    env.close()
