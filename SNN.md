# SNN Architecture & Implementation Guide
*For the SNN-Soft-Body-Evolution Project*

## 1. Core Philosophy
The Spiking Neural Network (SNN) will act as a Central Pattern Generator (CPG) to drive the soft-body creature. Because we are evolving this brain from scratch using **NEAT (NeuroEvolution of Augmenting Topologies)**, we will not use standard deep learning frameworks (like PyTorch) or specific SNN libraries (like snnTorch). 

Instead, the SNN will be implemented using a **high-performance, vectorized matrix approach** (pure NumPy or a simple C++ binding) to ensure lightning-fast execution during the Genetic Algorithm (GA) evaluation.

---

## 2. Input and Output Mapping (Bridging Physics & SNN)

### Inputs (Sensors)
The environment provides continuous physics data. This data must be encoded into discrete spikes for the input neurons.
*   **Method:** We will use **Direct Current Injection** (or Rate Encoding). 
*   **Implementation:** The physics engine calculates the **Percentage Deformation** of every spring: `Ratio = |L_current - L_rest| / L_rest`. 
*   This ratio is multiplied by a scaling factor `k` and injected directly as voltage into the membrane potential of the designated "Spring Sensor Neuron" on every tick. The harder a spring is stretched, the faster its sensor neuron fires spikes into the brain.

### Outputs (Actuators)
Spikes are instantaneous, but muscles need to contract smoothly to prevent physics glitches.
*   **Method:** We will use a **Leaky Integrator** on the output side.
*   **Implementation:** Every output neuron is paired with a specific spring. The spring maintains an `Activation Level` (0.0 to 1.0). 
*   When a spike arrives, it *integrates* (adds `+0.2` to the Activation Level). Every tick, the Activation Level *leaks* (multiplies by `0.9`). 
*   The physics engine uses this smooth Activation Level to dynamically shrink the spring's `rest_length`.

---

## 3. The Neuron Model (LIF)
We will use the **Leaky Integrate-and-Fire (LIF)** model.
Every simulation tick, the voltage vector $V$ (representing all neurons) undergoes:
1.  **Leak:** $V = V \times leak\_factor$ (e.g., 0.95).
2.  **Integrate:** Add incoming voltage from the environment (sensors) and from incoming spikes (matrix multiplication).
3.  **Fire:** If any neuron $V_i > Threshold$ (e.g., 1.0), it fires a spike, and its voltage resets $V_i = 0.0$.

---

## 4. Integrating NEAT (NeuroEvolution of Augmenting Topologies)
Because we are using NEAT, the brain's topology (size and shape) is dynamic. The genome and the simulation are split into two phases:

### Phase 1: The Genotype (Evolution)
The genome is a list of Connection Genes. Because this is an SNN, we extend the standard NEAT gene to include **Synaptic Delays**.
*   **Gene Structure:** `[In_Node, Out_Node, Weight, Delay, Enabled, Innov_ID]`
*   **Mutations:** The GA can add nodes, add connections, tweak weights, and tweak delay times.

### Phase 2: The Phenotype (Simulation)
When it's time to test a creature, the list of genes is compiled into static arrays/matrices for incredibly fast simulation.

---

## 5. The Fast Simulation Architecture (The Matrix Approach)
To avoid the slowness of Python `for`-loops and dictionaries, the genome is compiled into Sparse Matrices.

### The Matrices
1.  **State Vector ($V$)**: A 1D array holding the membrane potential of every neuron.
2.  **Weight Matrix ($W$)**: A 2D sparse matrix encoding the synaptic strengths.
3.  **Delay Matrix ($D$)**: A 2D matrix encoding the transmission delays (in discrete time steps).

### Handling Delays (The Ring Buffer)
To implement axonal delays without slow queues, we use a **Spike History Ring Buffer**.
*   We maintain a circular 2D array of past spikes. If the maximum delay is $T_{max} = 20$ ticks, the buffer holds the binary spike vectors for the last 20 ticks.
*   During an update, the Delay Matrix ($D$) is used to index into the Ring Buffer to fetch the correct historical spikes. 
*   We then perform a Sparse Matrix-Vector multiplication (SpMV) with the Weight Matrix to calculate the incoming voltage for all neurons in one single mathematical sweep.

### Execution Speed
By storing neuron states as flat arrays and using SpMV, the simulation can evaluate hundreds of creature brains simultaneously in fractions of a second using just NumPy, or even faster if Developer A ports the `integrate()` function to a small C++ module exposed via `pybind11`.
