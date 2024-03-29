#  Copyright (c) 2024. Kromek Group Ltd.
import random

from src.environments import Environment
from src.lifeforms import LifeForm


class Simulation:
    def __init__(self, size, num_life_forms):
        self.environment = Environment(size)
        self.life_forms = [LifeForm(species=random.choice(["SpeciesA", "SpeciesB"]),
                                    x=random.randint(0, size-1),
                                    y=random.randint(0, size-1))
                           for _ in range(num_life_forms)]
        # Initialize the simulation with specified size and number of life forms

    def run(self, num_steps):
        # Run the simulation for a specified number of time steps
        for _ in range(num_steps):
            self.environment.update()
            for life_form in self.life_forms:
                life_form.move(random.choice(["up", "down", "left", "right"]))
                life_form.eat(self.environment)
                life_form.reproduce()
            # Update life forms and environment for each time step

if __name__ == "__main__":
    simulation = Simulation(size=10, num_life_forms=50)
    simulation.run(num_steps=100)