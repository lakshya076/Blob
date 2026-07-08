class Integrator:
    def integrate(self, mass, dt, gravity):
        raise NotImplementedError

class SemiImplicitEuler(Integrator):
    def integrate(self, mass, dt: float, gravity: float):
        mass.fy += mass.mass * gravity

        ax = mass.fx / mass.mass
        ay = mass.fy / mass.mass

        mass.vx += ax * dt
        mass.vy += ay * dt

        mass.x += mass.vx * dt
        mass.y += mass.vy * dt
