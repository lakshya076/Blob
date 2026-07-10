import random
import numpy as np
from snn import SpikingNetwork

class NodeGene:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type # 'BIAS', 'INPUT', 'OUTPUT', 'HIDDEN'

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, delay, innov_id, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.delay = delay
        self.innov_id = innov_id
        self.enabled = enabled

class InnovationTracker:
    def __init__(self):
        self.current_innov_id = 0
        self.history = {}

    def get_innov_id(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self.history:
            self.history[key] = self.current_innov_id
            self.current_innov_id += 1
        return self.history[key]
    
tracker = InnovationTracker()

class Genome:
    def __init__(self, num_inputs, num_outputs, init_connections=True):
        self.nodes = {}
        self.connections = []
        self.next_node_id = 0

        # Create Bias (Node 0)
        self.nodes[self.next_node_id] = NodeGene(self.next_node_id, 'BIAS')
        bias_id = self.next_node_id
        self.next_node_id += 1

        # Create Inputs
        input_ids = []
        for _ in range(num_inputs):
            self.nodes[self.next_node_id] = NodeGene(self.next_node_id, 'INPUT')
            input_ids.append(self.next_node_id)
            self.next_node_id += 1

        # Create Outputs
        output_ids = []
        for _ in range(num_outputs):
            self.nodes[self.next_node_id] = NodeGene(self.next_node_id, 'OUTPUT')
            output_ids.append(self.next_node_id)
            self.next_node_id += 1
            
        if init_connections:
            # 1. Spawn a random seed hidden layer (3 to 5 nodes)
            num_hidden = random.randint(3, 5)
            hidden_ids = []
            for _ in range(num_hidden):
                self.nodes[self.next_node_id] = NodeGene(self.next_node_id, 'HIDDEN')
                hidden_ids.append(self.next_node_id)
                self.next_node_id += 1
                
            # 2. Randomly connect Inputs -> Hidden Layer
            for in_id in [bias_id] + input_ids:
                # Connect each input to a random hidden node
                h_id = random.choice(hidden_ids)
                w = random.uniform(-1.0, 1.0); d = random.randint(1, 5)
                self.connections.append(ConnectionGene(in_id, h_id, w, d, tracker.get_innov_id(in_id, h_id)))

            # 3. Randomly connect Hidden Layer -> Outputs
            for out_id in output_ids:
                h_id = random.choice(hidden_ids)
                w = random.uniform(-1.0, 1.0); d = random.randint(1, 5)
                self.connections.append(ConnectionGene(h_id, out_id, w, d, tracker.get_innov_id(h_id, out_id)))

    def distance(self, other, c1=1.0, c2=1.0, c3=0.4):
        # Calculates genetic distance for Speciation
        p1 = {c.innov_id: c for c in self.connections if c.enabled}
        p2 = {c.innov_id: c for c in other.connections if c.enabled}
        if not p1 and not p2: return 0.0
        
        m1 = max(p1.keys()) if p1 else 0
        m2 = max(p2.keys()) if p2 else 0
        
        disjoint = excess = matching = 0
        weight_diff = 0.0
        
        all_innovs = set(p1.keys()).union(set(p2.keys()))
        for innov in all_innovs:
            if innov in p1 and innov in p2:
                matching += 1
                weight_diff += abs(p1[innov].weight - p2[innov].weight)
            elif innov in p1:
                if innov > m2: excess += 1
                else: disjoint += 1
            elif innov in p2:
                if innov > m1: excess += 1
                else: disjoint += 1
                
        N = max(len(p1), len(p2))
        if N < 20: N = 1 # Normalize for small genomes
        
        return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * (weight_diff / max(1, matching)))

    def mutate_weight(self, power=0.5):
        if not self.connections: return
        conn = random.choice(self.connections)
        conn.weight += random.uniform(-power, power)
        conn.weight = max(-1.0, min(1.0, conn.weight))

    def mutate_delay(self, max_delay):
        if not self.connections: return
        conn = random.choice(self.connections)
        conn.delay += random.randint(-2, 2)
        conn.delay = max(1, min(max_delay, conn.delay))

    def mutate_add_connection(self, max_delay):
        node_ids = list(self.nodes.keys())
        for _ in range(20):
            in_node_id = random.choice(node_ids)
            out_node_id = random.choice(node_ids)
            
            if self.nodes[out_node_id].type in ["INPUT", "BIAS"]:
                continue
                
            # BAN direct Input -> Output connections
            if self.nodes[in_node_id].type in ["INPUT", "BIAS"] and self.nodes[out_node_id].type == "OUTPUT":
                continue
            exists = any(c.in_node == in_node_id and c.out_node == out_node_id for c in self.connections)
            if not exists:
                w = random.uniform(-1.0, 1.0)
                d = random.randint(1, max_delay)
                i_id = tracker.get_innov_id(in_node_id, out_node_id)
                self.connections.append(ConnectionGene(in_node_id, out_node_id, w, d, i_id))
                return

    def mutate_add_node(self):
        if not self.connections: return
        conn = random.choice(self.connections)
        conn.enabled = False
        new_id = self.next_node_id
        self.nodes[new_id] = NodeGene(new_id, 'HIDDEN')
        self.next_node_id += 1
        
        i1 = tracker.get_innov_id(conn.in_node, new_id)
        i2 = tracker.get_innov_id(new_id, conn.out_node)
        self.connections.append(ConnectionGene(conn.in_node, new_id, 1.0, 1, i1))
        self.connections.append(ConnectionGene(new_id, conn.out_node, conn.weight, conn.delay, i2))

    def mutate_toggle_connection(self):
        # Structural Culling
        if not self.connections: return
        conn = random.choice(self.connections)
        conn.enabled = not conn.enabled

    def build_phenotype(self, leak_factor=0.9, threshold=1.0, max_biological_delay=0.5, refractory_period=0.05):
        total = len(self.nodes)
        weights = np.zeros((total, total))
        delays = np.ones((total, total), dtype=int)
        
        node_map = {n_id: i for i, n_id in enumerate(self.nodes.keys())}
        
        for c in self.connections:
            if c.enabled:
                r, col = node_map[c.in_node], node_map[c.out_node]
                weights[r, col] = c.weight
                delays[r, col] = c.delay
        return SpikingNetwork(weights=weights, delays=delays, leak_factor=leak_factor, threshold=threshold, max_biological_delay=max_biological_delay, refractory_period=refractory_period)

def crossover(p1, p2):
    child = Genome(0, 0, init_connections=False)
    child.nodes = {k: NodeGene(v.id, v.type) for k, v in p1.nodes.items()}
    child.next_node_id = p1.next_node_id
    
    c1 = {c.innov_id: c for c in p1.connections}
    c2 = {c.innov_id: c for c in p2.connections}
    
    for innov in set(c1.keys()).union(set(c2.keys())):
        if innov in c1 and innov in c2:
            chosen = random.choice([c1[innov], c2[innov]])
            child.connections.append(ConnectionGene(chosen.in_node, chosen.out_node, chosen.weight, chosen.delay, chosen.innov_id, chosen.enabled))
        elif innov in c1:
            c = c1[innov]
            child.connections.append(ConnectionGene(c.in_node, c.out_node, c.weight, c.delay, c.innov_id, c.enabled))
    return child

class Species:
    def __init__(self, rep):
        self.rep = rep
        self.members = []
        self.adjusted_fitness = 0.0

class Population:
    def __init__(self, size, num_inputs, num_outputs):
        self.size = size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genomes = [Genome(num_inputs, num_outputs) for _ in range(size)]
        self.species = []
        
    def evolve(self, fitness_scores, max_delay, mut_rates=None, compat_thresh=3.0, c1=1.0, c2=1.0, c3=0.4, weight_power=0.5):
        if mut_rates is None:
            mut_rates = {'weight': 0.8, 'delay': 0.8, 'add_conn': 0.1, 'add_node': 0.05, 'toggle_conn': 0.1}
            
        # 1. Speciate the population
        self.species = []
        for g, f in zip(self.genomes, fitness_scores):
            g.fitness = f
            placed = False
            for s in self.species:
                if g.distance(s.rep, c1=c1, c2=c2, c3=c3) < compat_thresh: # Compatibility threshold
                    s.members.append(g)
                    placed = True
                    break
            if not placed:
                s = Species(g)
                s.members.append(g)
                self.species.append(s)
                
        # 2. Adjusted Fitness Sharing (Protecting Innovation)
        new_pop = []
        for s in self.species:
            for g in s.members:
                g.adj_fitness = g.fitness / len(s.members)
            s.members.sort(key=lambda x: x.adj_fitness, reverse=True)
            s.adjusted_fitness = sum(g.adj_fitness for g in s.members)
            
            # Elitism per species (Save the best of each niche)
            if len(s.members) > 0:
                new_pop.append(s.members[0])
                
        # 3. Breed the next generation
        self.species.sort(key=lambda x: x.adjusted_fitness, reverse=True)
        total_adj = sum(s.adjusted_fitness for s in self.species if s.adjusted_fitness > 0)
        
        while len(new_pop) < self.size:
            # Pick a species based on fitness
            if total_adj > 0:
                r = random.uniform(0, total_adj)
                curr = 0
                for s in self.species:
                    curr += s.adjusted_fitness
                    if curr >= r:
                        chosen_species = s
                        break
            else:
                chosen_species = random.choice(self.species)
                
            p1 = random.choice(chosen_species.members)
            p2 = random.choice(chosen_species.members)
            child = crossover(p1, p2)
            
            # Mutations
            if random.random() < mut_rates['weight']: child.mutate_weight(power=weight_power)
            if random.random() < mut_rates.get('delay', 0.8): child.mutate_delay(max_delay)
            if random.random() < mut_rates.get('add_conn', 0.1): child.mutate_add_connection(max_delay)
            if random.random() < mut_rates.get('add_node', 0.05): child.mutate_add_node()
            if random.random() < mut_rates.get('toggle_conn', 0.1): child.mutate_toggle_connection()
            new_pop.append(child)
            
        self.genomes = new_pop