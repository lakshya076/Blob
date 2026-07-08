import random
import numpy as np
from snn import SpikingNetwork

class NodeGene:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type # 'INPUT', 'OUTPUT', 'HIDDEN'

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
        self.history = {} # Maps (in_node, out_node) -> innov_id

    def get_innov_id(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self.history:
            self.history[key] = self.current_innov_id
            self.current_innov_id += 1
        return self.history[key]
    
tracker = InnovationTracker()

class Genome:
    def __init__(self, num_inputs, num_outputs):
        self.nodes = {}
        self.connections = []
        self.next_node_id = 0

        for _ in range(num_inputs):
            self.nodes[self.next_node_id] = NodeGene(self.next_node_id, 'INPUT')
            self.next_node_id += 1

        for _ in range(num_outputs):
            self.nodes[self.next_node_id] = NodeGene(self.next_node_id, 'OUTPUT')
            self.next_node_id += 1

    def mutate_weight(self):
        if not self.connections: return
        conn = random.choice(self.connections)
        conn.weight += random.uniform(-0.5, 0.5)
        conn.weight = max(-1.0, min(1.0, conn.weight)) # Clamp

    def mutate_delay(self, max_delay):
        if not self.connections: return
        conn = random.choice(self.connections)
        conn.delay += random.randint(-2, 2)
        conn.delay = max(1, min(max_delay, conn.delay)) # Clamp

    def mutate_add_connection(self, max_delay):
        node_ids = list(self.nodes.keys())
        for _ in range(20):
            in_node_id = random.choice(node_ids)
            out_node_id = random.choice(node_ids)

            if self.nodes[out_node_id].type == "INPUT":
                continue

            connection_exists = False
            for conn in self.connections:
                if conn.in_node == in_node_id and conn.out_node == out_node_id:
                    connection_exists = True
                    break
            
            if connection_exists:
                continue
            break
        else:
            return
    
        weight = random.uniform(-1.0, 1.0)
        delay = random.randint(1, max_delay)
        innov_id = tracker.get_innov_id(in_node_id, out_node_id)
        new_conn = ConnectionGene(in_node_id, out_node_id, weight, delay, innov_id)
        self.connections.append(new_conn)
            
    def mutate_add_node(self):
        if not self.connections: return
        conn = random.choice(self.connections)
        conn.enabled = False

        new_node_id = self.next_node_id
        self.nodes[new_node_id] = NodeGene(new_node_id, 'HIDDEN')
        self.next_node_id += 1

        in_to_new_id = tracker.get_innov_id(conn.in_node, new_node_id)
        new_to_out_id = tracker.get_innov_id(new_node_id, conn.out_node)

        self.connections.append(ConnectionGene(conn.in_node, new_node_id, 1.0, 1, in_to_new_id))
        self.connections.append(ConnectionGene(new_node_id, conn.out_node, conn.weight, conn.delay, new_to_out_id))

    def build_phenotype(self):
        total_nodes = len(self.nodes)
        weights = np.zeros((total_nodes, total_nodes))
        delays = np.ones((total_nodes, total_nodes), dtype=int)

        node_map = {node_id: index for index, node_id in enumerate(self.nodes.keys())}

        for conn in self.connections:
            if conn.enabled:
                row = node_map[conn.in_node]
                col = node_map[conn.out_node]
                weights[row, col] = conn.weight
                delays[row, col] = conn.delay

        return SpikingNetwork(weights=weights, delays=delays)

def crossover(parent1, parent2):
    child = Genome(0, 0)
    # Inherit nodes from parent 1 (assumed fitter)
    child.nodes = {k: NodeGene(v.id, v.type) for k, v in parent1.nodes.items()}
    child.next_node_id = parent1.next_node_id
    
    p1_conns = {c.innov_id: c for c in parent1.connections}
    p2_conns = {c.innov_id: c for c in parent2.connections}
    all_innovs = set(p1_conns.keys()).union(set(p2_conns.keys()))
    
    for innov in all_innovs:
        if innov in p1_conns and innov in p2_conns:
            chosen = random.choice([p1_conns[innov], p2_conns[innov]])
            child.connections.append(ConnectionGene(chosen.in_node, chosen.out_node, chosen.weight, chosen.delay, chosen.innov_id, chosen.enabled))
        elif innov in p1_conns:
            c = p1_conns[innov]
            child.connections.append(ConnectionGene(c.in_node, c.out_node, c.weight, c.delay, c.innov_id, c.enabled))
            
    return child

class Population:
    def __init__(self, size, num_inputs, num_outputs):
        self.size = size
        self.genomes = [Genome(num_inputs, num_outputs) for _ in range(size)]
        
    def evolve(self, fitness_scores, max_delay):
        paired = list(zip(self.genomes, fitness_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        # Elitism: keep top 20%
        num_elites = max(1, int(self.size * 0.2))
        elites = [g for g, f in paired[:num_elites]]
        
        new_pop = []
        new_pop.extend(elites)
        
        while len(new_pop) < self.size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = crossover(p1, p2)
            
            # Mutate child
            if random.random() < 0.8: child.mutate_weight()
            if random.random() < 0.8: child.mutate_delay(max_delay)
            if random.random() < 0.1: child.mutate_add_connection(max_delay)
            if random.random() < 0.05: child.mutate_add_node()
            
            new_pop.append(child)
            
        self.genomes = new_pop