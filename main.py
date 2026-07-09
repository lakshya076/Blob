import time
import numpy as np
import pickle
import os
import sys
import shutil
from physics import SoftBodySimulation
from evolution import Population
from hyperparameters import *
import multiprocessing

# Simulation time in seconds
SIMULATION_TIME = 5.0
PHYSICS_DT = 0.016 # ~60 fps

def evaluate_genome(genome, leak_factor=0.9, threshold=1.0, energy_penalty=0.0005, max_biological_delay=0.5):
    # This function evaluates a single brain's fitness
    brain = genome.build_phenotype(leak_factor=leak_factor, threshold=threshold, max_biological_delay=max_biological_delay)
    sim = SoftBodySimulation()
    start_x = np.mean([m.x for m in sim.masses])
    
    ticks = int(SIMULATION_TIME / PHYSICS_DT)
    for _ in range(ticks):
        sim.update(PHYSICS_DT, brain)
        
    # 4. Track ending center of mass
    end_x = np.mean([m.x for m in sim.masses])
    
    # 5. Fitness is distance traveled MINUS an energy penalty
    # This prevents the "screaming network" local minima
    total_spikes = np.sum(brain.buffer)
    distance = end_x - start_x
    fitness = distance - (total_spikes * energy_penalty)
    return fitness, distance

def train_mode():
    dummy_sim = SoftBodySimulation()
    num_springs = len(dummy_sim.springs)
    
    num_inputs = num_springs
    num_outputs = num_springs
    
    pop_input = input("Enter Population Size [default 50]: ")
    POPULATION_SIZE = int(pop_input) if pop_input.isdigit() else 50
    
    gen_input = input("Enter Number of Generations [default 100]: ")
    GENERATIONS = int(gen_input) if gen_input.isdigit() else 100
    MAX_DELAY = int(MAX_BIOLOGICAL_DELAY / PHYSICS_DT) # Use centralized setting # Max biological delay of 0.5s

    print(f"Initializing Population of {POPULATION_SIZE} with {num_inputs} Inputs and {num_outputs} Outputs...")
    pop = Population(POPULATION_SIZE, num_inputs, num_outputs)

    pool = multiprocessing.Pool()
    
    if os.path.exists("saved_brains"):
        shutil.rmtree("saved_brains")
    os.makedirs("saved_brains")

    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen+1} ---")
        start_time = time.time()
        
        # Evaluate using our centralized hyperparameters
        args_list = [(g, LEAK_FACTOR, THRESHOLD, ENERGY_PENALTY, MAX_BIOLOGICAL_DELAY) for g in pop.genomes]
        results = pool.starmap(evaluate_genome, args_list)
        
        fitness_scores = [r[0] for r in results]
        distances = [r[1] for r in results]
        
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_distance = distances[best_idx]
        best_genome = pop.genomes[best_idx]
        avg_fitness = np.mean(fitness_scores)
        
        print(f"Best Fitness: {best_fitness:.4f} (Distance: {best_distance:.4f})")
        print(f"Avg Fitness: {avg_fitness:.4f}")
        print(f"Time Taken: {time.time() - start_time:.2f}s")
        
        # Save the best brain of this generation
        with open(f"saved_brains/gen_{gen+1}.pkl", "wb") as f:
            pickle.dump(best_genome, f)
            
        print(f"Saved Best Brain to saved_brains/gen_{gen+1}.pkl")
        
        # Breed the next generation using our centralized hyperparameters
        pop.evolve(
            fitness_scores, 
            MAX_DELAY,
            mut_rates=MUTATION_RATES,
            compat_thresh=COMPATIBILITY_THRESHOLD,
            c1=C1, c2=C2, c3=C3,
            weight_power=WEIGHT_MUTATION_POWER
        )

    print("\nEvolution Complete!")

def replay_mode():
    if not os.path.exists("saved_brains"):
        print("No 'saved_brains' directory found. You need to train first!")
        return

    files = [f for f in os.listdir("saved_brains") if f.endswith('.pkl')]
    if not files:
        print("No saved brains found. Train first!")
        return

    print("\nAvailable Generations:")
    for f in sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0])):
        print(f" - {f}")

    choice = input("\nEnter the generation number you want to watch (e.g., 1): ")
    
    if choice.isdigit():
        choice = f"gen_{choice}.pkl"
    elif not choice.endswith('.pkl'):
        choice = f"{choice}.pkl"
        
    path = os.path.join("saved_brains", choice)

    if not os.path.exists(path):
        print("File not found.")
        return

    with open(path, "rb") as f:
        best_genome = pickle.load(f)

    # Compile the brain using the current hyperparameters
    brain = best_genome.build_phenotype(leak_factor=LEAK_FACTOR, threshold=THRESHOLD, max_biological_delay=MAX_BIOLOGICAL_DELAY)
    sim = SoftBodySimulation()

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"Replay: {choice}")
    clock = pygame.time.Clock()

    SCALE = 50.0
    OFFSET_X = 100
    OFFSET_Y = 400

    # --- Pre-calculate SNN visual layout ---
    node_positions = []
    num_inputs = len(sim.springs) + 1 # +1 for the Bias Node
    num_outputs = len(sim.springs)
    num_hidden = brain.N - num_inputs - num_outputs
    
    for i in range(brain.N):
        if i < num_inputs:
            x = 250
            y = 30 + (i + 0.5) * (150.0 / max(1, num_inputs))
        elif i < num_inputs + num_outputs:
            out_idx = i - num_inputs
            x = 550
            y = 30 + (out_idx + 0.5) * (150.0 / max(1, num_outputs))
        else:
            hid_idx = i - num_inputs - num_outputs
            # Stagger hidden nodes slightly in the middle (X: 350 to 450)
            x = 400 + ((i * 37) % 100) - 50 
            y = 30 + (hid_idx + 0.5) * (150.0 / max(1, num_hidden))
        node_positions.append((int(x), int(y)))
        
    # Pre-find connections to draw faint lines
    conns = []
    for r in range(brain.N):
        for c in range(brain.N):
            if brain.weights[r, c] != 0:
                conns.append((r, c))
    # ---------------------------------------

    running = True
    print(f"Playing {choice}... Close the PyGame window to return to the menu.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # 1. Update Physics
        sim.update(PHYSICS_DT, brain)
        
        # 2. Draw
        screen.fill((20, 20, 20))
        pygame.draw.line(screen, (100, 255, 100), (0, OFFSET_Y), (800, OFFSET_Y), 2)
        
        for spring in sim.springs:
            x1 = int(spring.m1.x * SCALE + OFFSET_X)
            y1 = int(spring.m1.y * SCALE + OFFSET_Y)
            x2 = int(spring.m2.x * SCALE + OFFSET_X)
            y2 = int(spring.m2.y * SCALE + OFFSET_Y)
            
            color = (255, 50, 50) if spring.activation > 0.1 else (200, 200, 200)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)
            
        for mass in sim.masses:
            x = int(mass.x * SCALE + OFFSET_X)
            y = int(mass.y * SCALE + OFFSET_Y)
            pygame.draw.circle(screen, (100, 150, 255), (x, y), 6)
            
        # 3. Draw the Brain
        # Draw wires first (so they are under nodes)
        for r, c in conns:
            pygame.draw.line(screen, (60, 60, 60), node_positions[r], node_positions[c], 1)
            
        # Draw Nodes
        last_spikes = brain.buffer[(brain.time_pointer - 1) % brain.T_max]
        for i in range(brain.N):
            # Bright green if firing, white if silent
            color = (50, 255, 50) if last_spikes[i] > 0 else (255, 255, 255)
            pygame.draw.circle(screen, color, node_positions[i], 4)
            
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()

if __name__ == '__main__':
    while True:
        print("\n=== SNN Soft-Body Evolution ===")
        print("1. Train New Population")
        print("2. Replay Saved Generation")
        print("3. Exit")
        
        choice = input("Select an option (1/2/3): ")
        if choice == '1':
            train_mode()
        elif choice == '2':
            replay_mode()
        elif choice == '3':
            sys.exit(0)
        else:
            print("Invalid choice.")
