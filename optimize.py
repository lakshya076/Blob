import optuna
import multiprocessing
import numpy as np
from main import evaluate_genome, PHYSICS_DT
from evolution import Population
from physics import SoftBodySimulation

def objective(trial):
    # 1. Suggest SNN Hyperparameters
    leak_factor = trial.suggest_float("leak_factor", 0.5, 0.99)
    threshold = trial.suggest_float("threshold", 0.1, 2.0)
    max_biological_delay = trial.suggest_float("max_biological_delay", 0.1, 1.5)
    
    # 2. Suggest Scoring Hyperparameters
    energy_penalty = trial.suggest_float("energy_penalty", 1e-5, 1e-3, log=True)
    
    # 3. Suggest GA Speciation Hyperparameters
    compat_thresh = trial.suggest_float("compat_thresh", 1.5, 4.5)
    c1 = trial.suggest_float("c1", 0.5, 2.0)
    c2 = trial.suggest_float("c2", 0.5, 2.0)
    c3 = trial.suggest_float("c3", 0.1, 1.0)
    
    # 4. Suggest Mutation Hyperparameters
    weight_power = trial.suggest_float("weight_power", 0.1, 1.0)
    mut_weight = trial.suggest_float("mut_weight", 0.4, 0.9)
    mut_delay = trial.suggest_float("mut_delay", 0.4, 0.9)
    mut_add_conn = trial.suggest_float("mut_add_conn", 0.05, 0.3)
    mut_add_node = trial.suggest_float("mut_add_node", 0.01, 0.1)
    
    mut_rates = {
        'weight': mut_weight,
        'delay': mut_delay,
        'add_conn': mut_add_conn,
        'add_node': mut_add_node
    }

    dummy_sim = SoftBodySimulation()
    num_inputs = len(dummy_sim.springs) + 1 # Include Bias
    num_outputs = len(dummy_sim.springs)
    
    POP_SIZE = 20
    GENS = 8
    MAX_DELAY = int(max_biological_delay / PHYSICS_DT)
    
    pop = Population(POP_SIZE, num_inputs, num_outputs)
    
    # We use the global pool initialized at the bottom to avoid leaking processes
    global pool
    
    best_overall_fitness = -float('inf')

    for gen in range(GENS):
        args_list = [(g, leak_factor, threshold, energy_penalty, max_biological_delay) for g in pop.genomes]
        results = pool.starmap(evaluate_genome, args_list)
        
        fitness_scores = [r[0] for r in results]
        current_best = max(fitness_scores)
        
        if current_best > best_overall_fitness:
            best_overall_fitness = current_best
            
        # Early Stopping: If after 3 generations the best fitness is still terrible, abort.
        if gen == 2 and best_overall_fitness <= 0.0:
            raise optuna.TrialPruned()
            
        pop.evolve(
            fitness_scores, 
            MAX_DELAY, 
            mut_rates=mut_rates, 
            compat_thresh=compat_thresh, 
            c1=c1, c2=c2, c3=c3, 
            weight_power=weight_power
        )

    return best_overall_fitness

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization...")
    # Initialize the multiprocessing pool globally so trials can reuse it
    pool = multiprocessing.Pool()
    
    # Create study to MAXIMIZE the score
    # The MedianPruner will automatically kill unpromising trials early!
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # Run optimization for 50 experiments
    study.optimize(objective, n_trials=50)
    
    print("\n=========================================")
    print("Optimization Finished!")
    print(f"Best Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    
    # Save to a Markdown file
    with open("optuna_results.md", "w") as f:
        f.write("# Optuna Hyperparameter Optimization Results\n\n")
        f.write(f"**Best Score:** `{study.best_value:.4f}`\n\n")
        f.write("### Best Hyperparameters\n")
        f.write("| Hyperparameter | Optimal Value |\n")
        f.write("|---|---|\n")
        
        for key, value in study.best_params.items():
            val_str = f"{value:.6f}" if isinstance(value, float) else str(value)
            print(f"  {key}: {val_str}")
            f.write(f"| `{key}` | `{val_str}` |\n")
            
    print("\nResults successfully saved to optuna_results.md")
    print("=========================================")
