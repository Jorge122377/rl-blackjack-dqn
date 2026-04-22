import torch
import gymnasium as gym

from dqn_agent import DQN, select_action
from utils import preprocess_state, ReplayBuffer

# Crear entorno
env = gym.make("Blackjack-v1")

# Modelo
model = DQN(input_dim=3, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Buffer
buffer = ReplayBuffer(capacity=1000)

# Parámetros
num_episodes = 300
epsilon = 1.0

for episode in range(num_episodes):

    obs, _ = env.reset()
    state = torch.tensor(preprocess_state(obs), dtype=torch.float32)

    done = False

    print(f"\n===== Episodio {episode+1} =====")

    while not done:

        # Elegir acción
        action = select_action(model, state, epsilon)

        # Ejecutar acción
        next_obs, reward, terminated, truncated, _ = env.step(action)

        next_state = torch.tensor(preprocess_state(next_obs), dtype=torch.float32)

        done = terminated or truncated

        # Guardar experiencia
        buffer.push(state, action, reward, next_state, done)

        # Entrenamiento (solo si hay suficientes datos)
        if len(buffer) >= 4:
            batch = buffer.sample(4)

            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for s, a, r, ns, d in batch:
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            dones = torch.tensor(dones)

            # Q actual
            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Q objetivo
            with torch.no_grad():
                next_q_values = model(next_states)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                target = rewards + 0.99 * max_next_q * (1 - dones.float())

            # Pérdida
            loss = torch.nn.functional.mse_loss(q_values, target)

            # Optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Loss:", loss.item())

        print("Estado:", state)
        print("Acción:", action)
        print("Recompensa:", reward)
        print("-----")

        # Avanzar estado
        state = next_state

    epsilon = max(0.01, epsilon * 0.995)
    print("Epsilon actual:", epsilon)

print("\nTamaño final del buffer:", len(buffer))

print("\n===== EVALUACIÓN DEL AGENTE =====")

eval_episodes = 10
eval_epsilon = 0.0

for episode in range(eval_episodes):
    obs, _ = env.reset()
    state = torch.tensor(preprocess_state(obs), dtype=torch.float32)

    done = False
    total_reward = 0

    print(f"\n--- Evaluación episodio {episode + 1} ---")

    while not done:
        action = select_action(model, state, eval_epsilon)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(preprocess_state(next_obs), dtype=torch.float32)

        done = terminated or truncated
        total_reward += reward

        print("Estado:", state)
        print("Acción elegida:", action)
        print("Recompensa:", reward)

        state = next_state

    print("Recompensa total del episodio:", total_reward)

env.close()