# SNN Soft-Body Evolution

A soft-body physics simulation built from scratch in Python, featuring evolutionary Spiking Neural Networks (SNNs) driven by a highly modified NEAT algorithm to control and propel a blob-like creature.

## Features
- **Custom Physics Engine**: Mass-spring soft-body physics using Semi-Implicit Euler integration, with gravity, ground collision, and friction.
- **Spiking Neural Networks (SNN)**: A continuous-time nervous system where neurons integrate voltage and spike to activate muscles.
- **Modified NEAT Evolution**: NeuroEvolution of Augmenting Topologies featuring speciation, structural culling, and delayed synaptic pathways.
- **Bayesian Hyperparameter Optimization**: Uses Optuna to find the perfect biological constants (refractory periods, leak factors) and GA mutation rates.
- **PyGame Visualization & Video Export**: Real-time rendering, interactive "Hand of God" grabbing, and automatic 60fps `.mp4` video rendering via OpenCV.

## Requirements

Ensure you have Python 3 installed. Install the dependencies using:

```bash
pip install -r requirements.txt
```

*(Dependencies: `pygame`, `numpy`, `optuna`, `opencv-python`)*

## Usage

Run the main script to access the interactive menu:

```bash
python main.py
```

### Menu Options:
1. **Train New Population**: Starts an evolutionary training loop using multiprocessing. The best brain from each generation is saved to the `saved_brains/` directory.
2. **Replay Saved Generation**: Loads a `.pkl` brain and renders its physics and neural activity live using PyGame. You can click and drag the creature around.
3. **Render Video (.mp4)**: Silently renders a chosen generation's PyGame output to a pristine 60fps `.mp4` file saved in `exported_videos/`.
4. **Exit**.

### Hyperparameter Optimization:
Run Optuna to find the mathematically perfect configuration for the SNN and the Genetic Algorithm:
```bash
python optimize.py
```
After optimization, it will generate an `optuna_results.md` report and optionally auto-apply the winning values directly into `hyperparameters.py`.

## Architecture Details

### 1. Physics Engine (`physics.py`)
- **Point Masses & Springs**: The creature is an interconnected mesh of masses and springs.
- **Muscle Actuation**: When a spring receives a neural spike, its `target_length` instantly contracts to 50% of its rest length. Without a spike, it attempts to return to 100% rest length.
- **Sensors (Inputs)**: The absolute Y-position (height) of every Point Mass is normalized and fed continuously into the neural network as sensory input, alongside a constant Bias signal of `1.0`.

### 2. Spiking Neural Network (`snn.py`)
- **Leaky Integrate-and-Fire**: Neurons accumulate incoming voltage. If they cross the `THRESHOLD`, they fire a spike (value `1.0`) and reset to `0.0`. Every tick, voltage decays based on the `LEAK_FACTOR`.
- **Axonal Delays**: Synapses carry a delay (1 to max ticks). Spikes are stored in a NumPy ring buffer (`buffer`) and only hit the target neuron after the delay elapses.
- **Refractory Period (Forced Silence)**: To prevent continuous "screaming" muscle spasms, neurons are forced to remain at `0.0` voltage for a strict `REFRACTORY_PERIOD` duration after firing.

### 3. Evolutionary Algorithm (`evolution.py`)
This project heavily modifies the NEAT (NeuroEvolution of Augmenting Topologies) architecture specifically for continuous control:
- **Seed Hidden Layer**: To prevent initial paralysis while avoiding direct input-to-muscle connections, Generation 0 spawns with 3-5 random Hidden Nodes. Inputs are randomly wired to the hidden layer, and the hidden layer to outputs.
- **Banned Direct Connections**: `Input -> Output` connections are explicitly blocked during mutation to prevent networks from falling into "screaming" local optima.
- **Structural Culling**: Features a unique `toggle_conn` mutation that can randomly disable an active connection, allowing the GA to easily shed obsolete synaptic pathways.
- **Standard NEAT Mutations**:
  - `mutate_weight`: Perturbs the weight of a random connection.
  - `mutate_delay`: Changes the biological delay of a synapse.
  - `mutate_add_connection`: Wires two disconnected nodes (respecting the direct-connection ban).
  - `mutate_add_node`: **Only** creates new hidden nodes by splitting an existing, functional connection, ensuring the new node is immediately integrated into the active pathway.
- **Speciation**: Genomes are clustered into species based on topological differences (Excess/Disjoint genes and weight differences) to protect new, fragile innovations.

## Configuration
All rules of the universe (SNN biology, GA mutation rates, Speciation math) are centralized in `hyperparameters.py`. 

## License
MIT License
