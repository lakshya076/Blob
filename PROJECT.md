# SNN-Soft-Body-Evolution

## 1. Project Explanation & Core Concepts

This project aims to evolve a Spiking Neural Network (SNN) that acts as a brain to control a 2D soft-body creature, enabling it to learn how to crawl or jump over time. 

### Spiking Neural Networks (SNNs)
Unlike traditional Artificial Neural Networks (ANNs) that pass continuous numbers through layers, SNNs communicate via discrete binary "spikes" over time, mimicking biological brains.
* **Leaky Integrate-and-Fire (LIF):** Each neuron has a "membrane potential". Incoming spikes increase this voltage. If no spikes arrive, the voltage "leaks" back down. If the voltage crosses a threshold, the neuron fires its own spike and resets.
* **Time as Data:** Because connections have "axonal delays", a spike might take 3 milliseconds to travel between nodes. This allows the network to naturally process time and rhythm, acting as a Central Pattern Generator (CPG) to drive cyclic motions like walking.

### Soft-Body Physics
The creature is modeled as a 2D mesh of point-masses connected by springs and dampers.
* **The Math:** We use Hooke's Law for spring forces ($F = -k(x - L_{rest})$) and apply a damping force ($F = -cv$) to prevent infinite oscillations.
* **Actuation:** The SNN controls the creature. When an output neuron spikes, it temporarily reduces the `rest_length` of a specific spring. The spring physically contracts, pulling two masses together.

### The Genetic Algorithm (GA)
Because SNN spikes are non-differentiable (they are instant events), we cannot easily use backpropagation. Instead, we use a Genetic Algorithm:
* A population of 20-50 random SNN genomes is generated.
* Each genome is placed in the physics simulation. The distance it travels is its "fitness" score.
* The best performers are selected, crossed over, and mutated to create the next generation.

---

## 2. Distribution of Work

The project is divided into four strict modules to allow two developers to work simultaneously. 

### Developer A: The Simulator (Physics & SNN)
**Files:** `physics.py` and `snn.py`
**Responsibilities:**
1. Build the mass-spring-damper physics engine using Semi-Implicit Euler integration.
2. Implement ground collision and friction logic.
3. Build the LIF neuron update loop and synaptic delay buffer.
4. Hook the SNN output spikes to the spring resting lengths.
5. Create a basic visualizer (e.g., Pygame) to see the body move.

### Developer B: The Optimizer (GA & Multiprocessing)
**Files:** `evolution.py` and `main.py`
**Responsibilities:**
1. Define the genome structure (a Python dictionary representing nodes and synapses).
2. Write the GA operations: tournament selection, crossover (swapping sub-graphs), and mutation (altering weights/delays).
3. Build the master loop in `main.py` that orchestrates generations.
4. Implement Python's `multiprocessing.Pool` so the population is evaluated in parallel across all CPU cores.

---

## 3. How to Achieve Zero-Blocking Parallel Development

To ensure both developers can work and test their code without waiting for the other to finish, you must use **Mock Stubs** inside your isolated testing blocks (`if __name__ == '__main__':`).

### Developer A's Isolated Testing
Developer A cannot test if their physics engine works until the SNN is built. 
**The Solution:** At the bottom of `physics.py`, create a `MockSpikingNeuralNetwork` that simply outputs a hardcoded alternating boolean pattern (e.g., using `math.sin(time)`). 
Developer A can run `python physics.py` standalone. The mock SNN will actuate the springs rhythmically, allowing Developer A to visually verify that contractions lead to movement and that the physics don't explode.

### Developer B's Isolated Testing
Developer B cannot test if the GA optimizes anything until the physics engine is built.
**The Solution:** Inside `main.py`, create a `MockSoftBodySimulation`. Instead of running physics, this mock class simply sums up all the "weights" inside the SNN genome and returns that as the fitness score.
Developer B can run `python main.py` standalone. The GA will naturally select and mutate genomes to maximize the weights. Developer B can verify that the multiprocessing pool is stable and the average fitness goes up every generation.

---

## 4. The 3-Hour Integration Plan

* **Hour 1 (0:00 - 1:00):** Dev A writes the mass-spring physics. Dev B writes the GA crossover/mutation logic. Both work offline using their mocks.
* **Hour 2 (1:00 - 2:00):** Dev A writes the LIF neuron dynamics. Dev B wires the multiprocessing pool in `main.py`.
* **Hour 3 (2:00 - 2:30):** Dev A connects the SNN spikes to the spring actuators and tests visually. Dev B ensures the GA successfully optimizes the mock fitness score.
* **Final Stretch (2:30 - 3:00):** Integration. In `main.py`, Dev B deletes the `MockSoftBodySimulation` and imports Dev A's real `SoftBodySimulation`. The system runs end-to-end.