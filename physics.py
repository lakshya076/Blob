from integrator import Integrator, SemiImplicitEuler
import math
import numpy as np

gravity = 9.81

class PointMass:
    def __init__(self, x, y, mass=1.0):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = 0
        self.vy = 0
        self.fx = 0
        self.fy = 0

    def apply_force(self, fx, fy):
        self.fx += fx
        self.fy += fy

    def update(self, dt, integrator: Integrator, ground_y=0.0, restitution=0.2, friction=0.8):
        integrator.integrate(self, dt, gravity)
        
        self.fx = 0.0
        self.fy = 0.0
        
        if self.y > ground_y:
            self.y = ground_y
            self.vy = -self.vy * restitution
            self.vx *= friction

        
class Spring:
    def __init__(self, mass1: PointMass, mass2: PointMass, k=100.0, damping=5.0):
        self.m1 = mass1
        self.m2 = mass2
        self.k = k
        self.damping = damping

        dx = self.m2.x - self.m1.x
        dy = self.m2.y - self.m1.y
        
        self.base_reset_length = math.sqrt(dx**2 + dy**2)
        self.rest_length = self.base_reset_length
        self.activation = 0.0

    def update(self, dt):
        if self.activation > 0:
            self.activation -= dt*2.0
        if self.activation < 0:
            self.activation = 0

        contraction_factor = 1.0 - (0.3*self.activation)
        self.rest_length = self.base_reset_length*contraction_factor

        dx = self.m2.x - self.m1.x
        dy = self.m2.y - self.m1.y
        
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist==0:
            return

        force_magnitude = self.k * (dist - self.rest_length)

        dvx = self.m2.vx - self.m1.vx
        dvy = self.m2.vy - self.m1.vy
        nx = dx / dist
        ny = dy / dist

        damping_force = self.damping * (dvx*nx + dvy*ny)

        total_force = force_magnitude + damping_force

        fx = total_force * nx
        fy = total_force * ny

        self.m1.apply_force(fx, fy)
        self.m2.apply_force(-fx, -fy)        


class SoftBodySimulation:
    def __init__(self):
        self.masses = list()
        self.springs = list()
        self.time = 0.0
        self.integrator = SemiImplicitEuler()
        self.build_worm()
    
    def build_worm(self):
        segments = 4
        width = 1.0
        height = 1.0
        
        for i in range(segments + 1):
            x = i * width
            self.masses.append(PointMass(x, -height, mass=1.0)) 
            self.masses.append(PointMass(x, 0, mass=1.0))       
            
        for i in range(segments):
            idx = i * 2
            top_left = self.masses[idx]
            bot_left = self.masses[idx+1]
            top_right = self.masses[idx+2]
            bot_right = self.masses[idx+3]
            
            self.springs.append(Spring(top_left, top_right))
            self.springs.append(Spring(bot_left, bot_right))
            self.springs.append(Spring(top_left, bot_left))
            self.springs.append(Spring(top_right, bot_right))
            
            self.springs.append(Spring(top_left, bot_right))
            self.springs.append(Spring(bot_left, top_right))


    def update(self, dt, snn):
        self.time += dt

        # 1. Gather sensor data (Percentage Deformation of each spring)
        num_springs = len(self.springs)
        sensor_data = np.zeros(snn.N)
        for i, spring in enumerate(self.springs):
            dx = spring.m2.x - spring.m1.x
            dy = spring.m2.y - spring.m1.y
            dist = math.sqrt(dx**2 + dy**2)
            # % deformation
            ratio = abs(dist - spring.base_reset_length) / spring.base_reset_length
            sensor_data[i] = ratio * 10.0 # Scaling factor

        # 2. Step the brain forward
        spike_signals = snn.step(inputs=sensor_data)

        # 3. Apply outputs to muscles
        # Outputs are indexed immediately after the inputs (0 to num_springs-1)
        for i in range(num_springs):
            output_idx = num_springs + i
            if output_idx < len(spike_signals) and spike_signals[output_idx] == 1.0:
                self.springs[i].activation = 1.0

        for i in self.springs:
            i.update(dt)

        for i in self.masses:
            i.update(dt, self.integrator)

            

class MockSpikingNeuralNetwork:
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    def get_outputs(self, time):
        outputs = []
        for i in range(self.num_outputs):
            phase = i * 0.5
            val = math.sin(time * 5.0 + phase)
            outputs.append(val > 0.5)

        return outputs


if __name__ == "__main__":
    import pygame
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Soft Body Physics Test")
    clock = pygame.time.Clock()
    
    # Initialize the simulation and the dummy brain
    sim = SoftBodySimulation()
    snn = MockSpikingNeuralNetwork(len(sim.springs))
    
    # Camera variables to map math coordinates to the screen
    SCALE = 50.0  # 1 math unit = 50 pixels
    OFFSET_X = 100
    OFFSET_Y = 400
    
    running = True
    while running:
        dt = 0.016  # Fixed timestep (~60 FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # 1. Update Physics
        sim.update(dt, snn)
        
        # 2. Draw Everything
        screen.fill((20, 20, 20)) # Dark gray background
        
        # Draw the ground line
        pygame.draw.line(screen, (100, 255, 100), (0, OFFSET_Y), (800, OFFSET_Y), 2)
        
        # Draw Springs
        for spring in sim.springs:
            # Map math coords to screen coords
            x1 = int(spring.m1.x * SCALE + OFFSET_X)
            y1 = int(spring.m1.y * SCALE + OFFSET_Y)
            x2 = int(spring.m2.x * SCALE + OFFSET_X)
            y2 = int(spring.m2.y * SCALE + OFFSET_Y)
            
            # Color the spring red if it is contracting (flexing), else white
            color = (255, 50, 50) if spring.activation > 0.1 else (200, 200, 200)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)
            
        # Draw Masses
        for mass in sim.masses:
            x = int(mass.x * SCALE + OFFSET_X)
            y = int(mass.y * SCALE + OFFSET_Y)
            pygame.draw.circle(screen, (100, 150, 255), (x, y), 6)
            
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()
