import numpy as np

dt = 0.1

class SpikingNetwork:

    def __init__(self, weights, delays):

        self.weights = weights
        self.delays = delays
        self.leak_factor = 0.9
        self.threshold = 1.0

        self.N = weights.shape[0]
        self.V = np.zeros(self.N)

        max_biological_delay = 0.5 # Time it takes for a signal to transfer from one end of the network to another
        self.T_max = int(max_biological_delay / dt)

        self.buffer = np.zeros((self.T_max, self.N))

        self.time_pointer = 0

    def step(self, inputs):

        self.V *= self.leak_factor
        self.V += inputs

        # Calculates row in ring buffer for each synapse
        history_indices = (self.time_pointer - self.delays) % self.T_max 

        source_nodes = np.arange(self.N)[:, None]

        arriving_spikes = self.buffer[history_indices, source_nodes]

        weighted_spikes = arriving_spikes * self.weights

        self.V += np.sum(weighted_spikes, axis = 0)

        # Find those that crossed threshold
        spikes = (self.V >= self.threshold).astype(float)

        self.V[spikes==1] = 0.0 # Reset any that fire

        self.buffer[self.time_pointer] = spikes # Save today's spikes

        self.time_pointer = (self.time_pointer + 1) % self.T_max

        return spikes
    
if __name__ == "__main__":
    # ---------------------------------------------------------
    # MOCK NEAT GENOME COMPILER
    # In the real project, Developer C's NEAT algorithm will 
    # generate these matrices dynamically as the brain grows.
    # ---------------------------------------------------------
    
    # N = 3 (Index 0: Input, Index 1: Hidden, Index 2: Output)
    
    # Weight Matrix (Row: Source, Col: Destination)
    # Node 0 connects to Node 1. Node 1 connects to Node 2.
    dummy_weights = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 0.0]
    ])
    
    # Delay Matrix (How many ticks the spike takes to arrive)
    # 5 ticks from 0 -> 1. And 5 ticks from 1 -> 2.
    dummy_delays = np.array([
        [0, 5, 0],
        [0, 0, 5],
        [0, 0, 0]
    ])
    
    # Instantiate your brain
    brain = SpikingNetwork(weights=dummy_weights, delays=dummy_delays)
    
    print("Starting SNN Simulation...")
    
    # Run the physics loop for 20 ticks
    for tick in range(20):
        # Create a blank array for environmental inputs
        sensor_data = np.zeros(brain.N)
        
        # Violently trigger the input sensor ONLY on tick 2
        if tick == 2:
            sensor_data[0] = 5.0 
            print(f"--- TICK 2: ENVIRONMENT POKED SENSOR 0 ---")
            
        # Step the brain forward one tick
        spikes = brain.step(inputs=sensor_data)
        
        # Check if the Hidden Node or Output Node fired
        if spikes[1] == 1.0:
            print(f"Tick {tick}: Hidden Node 1 fired!")
        if spikes[2] == 1.0:
            print(f"Tick {tick}: Output Node 2 fired! (MUSCLE CONTRACTS)")
