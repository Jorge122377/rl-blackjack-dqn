import gymnasium as gym
from utils import preprocess_state

# Crear entorno
env = gym.make("Blackjack-v1")

num_episodes = 5  # cantidad de episodios a ejecutar

for episode in range(num_episodes):
    obs, info = env.reset()

    print(f"\n===== Episodio {episode + 1} =====")
    print("Estado inicial:", obs)

    processed_obs = preprocess_state(obs)
    print("Estado procesado:", processed_obs)

    done = False

    while not done:
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        print("Acción:", action)
        print("Nuevo estado:", next_obs)

        processed_next_obs = preprocess_state(next_obs)
        print("Estado procesado:", processed_next_obs)

        print("Recompensa:", reward)
        print("------")

        done = terminated or truncated

    print("Resultado final del episodio:", reward)

env.close()