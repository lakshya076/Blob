# ==========================================
# SNN Hyperparameters
# ==========================================
LEAK_FACTOR = 0.589409      # How fast neuron voltage decays per tick
THRESHOLD = 1.177176        # Voltage required to fire a spike
MAX_BIOLOGICAL_DELAY = 0.152786 # Max time (in seconds) for a spike to travel across an axon

# ==========================================
# GA Scoring
# ==========================================
ENERGY_PENALTY = 0.000033 # Penalty deducted per spike (forces efficiency)

# ==========================================
# Mutation Probabilities & Power
# ==========================================
WEIGHT_MUTATION_POWER = 0.571782 # Max amount a weight can change in one mutation
MUTATION_RATES = {
    'weight': 0.496760,       # 80% chance to nudge a weight
    'delay': 0.599664,        # 80% chance to nudge a delay
    'add_conn': 0.108449,     # 10% chance to wire two random nodes
    'add_node': 0.035887     # 5% chance to insert a new Hidden neuron
}

# ==========================================
# Speciation Math (Protecting Innovation)
# ==========================================
COMPATIBILITY_THRESHOLD = 3.531190
C1 = 0.862722  # Importance of Excess genes
C2 = 0.510398  # Importance of Disjoint genes
C3 = 0.329403  # Importance of Weight differences
