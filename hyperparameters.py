# ==========================================
# SNN Hyperparameters
# ==========================================
LEAK_FACTOR = 0.678680      # How fast neuron voltage decays per tick
THRESHOLD = 0.715608        # Voltage required to fire a spike
MAX_BIOLOGICAL_DELAY = 1.362675 # Max time (in seconds) for a spike to travel across an axon
REFRACTORY_PERIOD = 0.008915 # Mandatory silence (in seconds) after a neuron fires


# ==========================================
# Mutation Probabilities & Power
# ==========================================
WEIGHT_MUTATION_POWER = 0.504478 # Max amount a weight can change in one mutation
MUTATION_RATES = {
    'weight': 0.880969,       # 80% chance to nudge a weight
    'delay': 0.510023,        # 80% chance to nudge a delay
    'add_conn': 0.121759,     # 10% chance to wire two random nodes
    'add_node': 0.037831,     # 5% chance to insert a new Hidden neuron
    'toggle_conn': 0.095458   # 10% chance to toggle a connection on/off (structural culling)
}

# ==========================================
# Speciation Math (Protecting Innovation)
# ==========================================
COMPATIBILITY_THRESHOLD = 3.446687
C1 = 0.848697  # Importance of Excess genes
C2 = 1.675094  # Importance of Disjoint genes
C3 = 0.540430  # Importance of Weight differences
