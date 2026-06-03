# RL Blackjack DQN

## Descripción del proyecto

Este proyecto implementa un agente de Aprendizaje por Refuerzo utilizando Deep Q-Network (DQN) para resolver el entorno Blackjack-v1 de Gymnasium.

El objetivo del agente es aprender automáticamente la mejor estrategia para jugar Blackjack a través de interacción con el entorno, recompensas y entrenamiento iterativo.

---

## Descripción de las acciones y observaciones

### Observaciones del entorno

El entorno Blackjack entrega un estado compuesto por tres valores:

- Suma actual de cartas del jugador
- Carta visible del dealer
- Existencia de un As utilizable (usable ace)

Ejemplo:

```python
(20, 10, 1)
```

Donde:

- 20 = suma de cartas del jugador
- 10 = carta visible del dealer
- 1 = existe un As utilizable

---

### Acciones disponibles

El agente puede realizar dos acciones:

- Acción 0 → Plantarse (Stick)
- Acción 1 → Pedir carta (Hit)

---

## Flujo lógico del entrenamiento

El entrenamiento del agente sigue el siguiente flujo:

1. El entorno Blackjack genera un estado inicial.
2. El estado es preprocesado y convertido a tensor.
3. El agente selecciona una acción usando estrategia epsilon-greedy.
4. El entorno responde con:
   - nuevo estado
   - recompensa
   - finalización del episodio
5. La experiencia se almacena en el Replay Buffer.
6. El agente toma batches aleatorios del buffer.
7. La red neuronal calcula valores Q.
8. Se calcula la pérdida (Loss).
9. El optimizador ajusta los pesos de la red neuronal.
10. El valor epsilon disminuye progresivamente para reducir exploración y aumentar explotación.

---

## Particularidades del ambiente Blackjack

El entorno Blackjack presenta desafíos importantes:

- El agente no conoce las cartas ocultas del dealer.
- Existe aleatoriedad en el reparto de cartas.
- Las recompensas son escasas:
  - +1 si gana
  - -1 si pierde
  - 0 en empate
- La decisión correcta depende del riesgo probabilístico.
- El entorno es episódico y termina rápidamente.

---

## Explicación de la red neuronal utilizada

Se implementó una red neuronal profunda usando PyTorch.

### Arquitectura:

- Entrada: 3 neuronas
  - suma del jugador
  - carta del dealer
  - usable ace

- Capas ocultas:
  - Capa lineal de 64 neuronas
  - ReLU
  - Segunda capa de 64 neuronas
  - ReLU

- Salida:
  - 2 neuronas
  - representan los valores Q de las acciones:
    - stick
    - hit

La red aprende a aproximar la función Q para seleccionar acciones óptimas.

---

## Técnicas implementadas

### Replay Buffer

Permite almacenar experiencias pasadas para reutilizarlas durante entrenamiento.

Cada experiencia contiene:

- estado
- acción
- recompensa
- siguiente estado
- done

---

### Estrategia epsilon-greedy

El agente balancea:

- exploración → probar acciones aleatorias
- explotación → usar conocimiento aprendido

El valor epsilon disminuye gradualmente:

```python
epsilon = max(0.01, epsilon * 0.995)
```

---

## Resultados del entrenamiento

El agente fue entrenado durante 300 episodios.

Resultados observados:

- Disminución progresiva de epsilon:
  - desde 1.0 hasta aproximadamente 0.22
- El Loss mostró estabilización parcial.
- El agente comenzó a tomar decisiones más consistentes.
- Durante evaluación logró episodios ganadores y empates.

Ejemplo de resultados:

```text
Loss: 0.4079
Epsilon actual: 0.222
```

---

## Evaluación del agente

Durante evaluación:

- El agente logró victorias en múltiples episodios.
- Aprendió a plantarse en estados favorables.
- En algunos casos siguió tomando decisiones subóptimas debido a la cantidad limitada de entrenamiento.

---

## Reflexión de los resultados

El proyecto permitió comprender cómo una red neuronal puede aprender estrategias mediante interacción y recompensas.

Se observó cómo el agente mejora gradualmente su comportamiento a medida que acumula experiencia.

También fue posible entender conceptos fundamentales de Deep Reinforcement Learning como:

- función Q
- exploración vs explotación
- Replay Buffer
- entrenamiento por batches
- optimización mediante descenso de gradiente

---

## Reflexión sobre las dificultades del proyecto

La parte más compleja del proyecto fue:

- comprender el flujo completo del entrenamiento DQN
- manejar correctamente tensores y tipos de datos en PyTorch
- implementar el cálculo de Loss
- entender cómo funciona el Replay Buffer
- depurar errores relacionados con dimensiones y dtypes

Además, fue necesario comprender detalladamente la interacción entre el entorno Gymnasium y la red neuronal.

---

## Estructura del proyecto

```text
src/
│
├── dqn_agent.py
├── env.py
├── train.py
└── utils.py
```

---

## Ejecución

```bash
python src/train.py
```