import optuna
import multiprocessing
import numpy as np
import re
from main import evaluate_genome, PHYSICS_DT
from evolution import Population
from physics import SoftBodySimulation

# ==========================================
# TEST SETTINGS (Configurable by you)
# ==========================================
OPT_POP_SIZE = 100
OPT_GENS = 100
OPT_TRIALS = 50
# ==========================================

def objective(trial):
    # 1. Suggest SNN Hyperparameters
    leak_factor = trial.suggest_float("leak_factor", 0.5, 0.99)
    threshold = trial.suggest_float("threshold", 0.1, 2.0)
    max_biological_delay = trial.suggest_float("max_biological_delay", 0.1, 1.5)
    refractory_period = trial.suggest_float("refractory_period", 0.0, 0.2)
    
    # 2. GA Speciation Hyperparameters
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
    mut_toggle_conn = trial.suggest_float("mut_toggle_conn", 0.01, 0.2)
    
    mut_rates = {
        'weight': mut_weight,
        'delay': mut_delay,
        'add_conn': mut_add_conn,
        'add_node': mut_add_node,
        'toggle_conn': mut_toggle_conn
    }

    dummy_sim = SoftBodySimulation()
    num_inputs = len(dummy_sim.springs) + 1 # Include Bias
    num_outputs = len(dummy_sim.springs)
    
    global OPT_POP_SIZE, OPT_GENS, pool
    MAX_DELAY = int(max_biological_delay / PHYSICS_DT)
    
    pop = Population(OPT_POP_SIZE, num_inputs, num_outputs)
    
    best_overall_fitness = -float('inf')

    for gen in range(OPT_GENS):
        args_list = [(g, leak_factor, threshold, max_biological_delay, refractory_period) for g in pop.genomes]
        results = pool.starmap(evaluate_genome, args_list)
        
        fitness_scores = [r[0] for r in results]
        current_best = max(fitness_scores)
        
        if current_best > best_overall_fitness:
            best_overall_fitness = current_best
            

            
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
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # Run optimization
    study.optimize(objective, n_trials=OPT_TRIALS)
    
    print("\n=========================================")
    print("Optimization Finished!")
    print(f"Best Score: {study.best_value:.4f}")
    
    # Save to a Markdown file
    with open("optuna_results.md", "w") as f:
        f.write("# Optuna Hyperparameter Optimization Results\n\n")
        f.write(f"**Best Score:** `{study.best_value:.4f}`\n\n")
        f.write("### Best Hyperparameters\n")
        f.write("| Hyperparameter | Optimal Value |\n")
        f.write("|---|---|\n")
        
        for key, value in study.best_params.items():
            val_str = f"{value:.6f}" if isinstance(value, float) else str(value)
            f.write(f"| `{key}` | `{val_str}` |\n")
            
    print("Results successfully saved to optuna_results.md")
    print("=========================================")
    
    # Auto-apply feature
    response = input("\nDo you want to automatically apply these optimal settings to hyperparameters.py? (y/n): ")
    if response.lower() == 'y':
        with open("hyperparameters.py", "r") as f:
            content = f.read()
            
        for key, value in study.best_params.items():
            if key == "max_biological_delay":
                content = re.sub(r"MAX_BIOLOGICAL_DELAY\s*=\s*[\d.e-]+", f"MAX_BIOLOGICAL_DELAY = {value:.6f}", content)
            elif key == "refractory_period":
                content = re.sub(r"REFRACTORY_PERIOD\s*=\s*[\d.e-]+", f"REFRACTORY_PERIOD = {value:.6f}", content)
            elif key == "leak_factor":
                content = re.sub(r"LEAK_FACTOR\s*=\s*[\d.e-]+", f"LEAK_FACTOR = {value:.6f}", content)
            elif key == "threshold":
                content = re.sub(r"THRESHOLD\s*=\s*[\d.e-]+", f"THRESHOLD = {value:.6f}", content)
            elif key == "weight_power":
                content = re.sub(r"WEIGHT_MUTATION_POWER\s*=\s*[\d.e-]+", f"WEIGHT_MUTATION_POWER = {value:.6f}", content)
            elif key == "compat_thresh":
                content = re.sub(r"COMPATIBILITY_THRESHOLD\s*=\s*[\d.e-]+", f"COMPATIBILITY_THRESHOLD = {value:.6f}", content)
            elif key == "c1":
                content = re.sub(r"C1\s*=\s*[\d.e-]+", f"C1 = {value:.6f}", content)
            elif key == "c2":
                content = re.sub(r"C2\s*=\s*[\d.e-]+", f"C2 = {value:.6f}", content)
            elif key == "c3":
                content = re.sub(r"C3\s*=\s*[\d.e-]+", f"C3 = {value:.6f}", content)
            elif key == "mut_weight":
                content = re.sub(r"'weight':\s*[\d.e-]+", f"'weight': {value:.6f}", content)
            elif key == "mut_delay":
                content = re.sub(r"'delay':\s*[\d.e-]+", f"'delay': {value:.6f}", content)
            elif key == "mut_add_conn":
                content = re.sub(r"'add_conn':\s*[\d.e-]+", f"'add_conn': {value:.6f}", content)
            elif key == "mut_add_node":
                content = re.sub(r"'add_node':\s*[\d.e-]+", f"'add_node': {value:.6f}", content)
            elif key == "mut_toggle_conn":
                content = re.sub(r"'toggle_conn':\s*[\d.e-]+", f"'toggle_conn': {value:.6f}", content)
                
        with open("hyperparameters.py", "w") as f:
            f.write(content)
        print("Success! All values in hyperparameters.py have been permanently updated!")
