# SNN Soft-Body Evolution

A soft-body physics simulation built from scratch in Python, featuring evolutionary Spiking Neural Networks (SNNs) that learn to control and propel a blob-like creature.

## Features

- **Custom Physics Engine**: Mass-spring based soft-body physics using Semi-Implicit Euler integration.
- **Spiking Neural Networks**: A simulated nervous system where neurons spike to activate spring contractions (muscles).
- **Evolutionary Algorithm**: Trains populations of SNN "brains" across generations to maximize distance traveled using a Genetic Algorithm.
- **PyGame Visualization**: Real-time rendering of the soft-body and a visual graph of the Spiking Neural Network's activity.
- **Hand of God**: Interact directly with the simulation in replay mode by clicking and dragging the creature's nodes with your mouse!

## Requirements

Ensure you have Python 3 installed. Install the dependencies using:

```bash
pip install -r requirements.txt
```

*(Dependencies include `pygame` and `numpy`)*

## Usage

Run the main script to access the simulation menu:

```bash
python main.py
```

### Menu Options:

1. **Train New Population**: Starts a new evolutionary training loop. You will be prompted to enter the population size and number of generations. The best brain from each generation will be saved to the `saved_brains/` directory.
2. **Replay Saved Generation**: Loads a previously trained `.pkl` brain from the `saved_brains/` directory and renders it using PyGame. 
   - *Tip*: Try clicking and dragging the nodes of the creature while it's moving!
3. **Exit**: Closes the application.

## Project Structure

- `physics.py`: Contains the core `PointMass`, `Spring`, and `SoftBodySimulation` classes.
- `integrator.py`: Implements the `SemiImplicitEuler` numerical integration.
- `snn.py` *(Assuming this handles the SNN logic)*: Handles the spiking neural network math and processing.
- `evolution.py` *(Assuming this handles the GA logic)*: Genetic algorithm implementation for breeding, crossover, and mutations.
- `main.py`: The entry point for the training loop and PyGame visualizer.
- `saved_brains/`: Directory containing pickled genome data from training sessions.

## Interactive Controls (Replay Mode)
- **Mouse Left Click & Drag**: Grab the closest physics node and throw it around the screen.

## License
MIT License
