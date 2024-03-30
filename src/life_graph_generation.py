import random

from src.attribute_enums import Size, Diet, Classification, Relationships
from src.environments import EnvironmentGraph

# Example usage:
if __name__ == "__main__":
    environment = EnvironmentGraph()

    # Add 4 life forms to the graph with attributes
    for i in range(1, 5):
        position = (random.uniform(0, 10), random.uniform(0, 10))  # Random position within a 10x10 grid
        speed = random.uniform(0, 10)
        size = random.choice(list(Size))
        diet = random.choice(list(Diet))
        classification = random.choice(list(Classification)[:3])
        environment.add_life_form(position, speed, size, diet, classification)

    # Add 4 resources to the graph with attributes
    for i in range(1, 5):
        position = (random.uniform(0, 10), random.uniform(0, 10))  # Random position within a 10x10 grid
        availability = random.uniform(0, 100)
        environment.add_resource(position, availability)

    # Add random interactions between entities
    interactions = [(Relationships.CONSUMES, 50), (Relationships.PREDATORPREY, 50)]

    for interaction, num in interactions:
        for _ in range(num):
            entity1_id = random.choice(list(environment.used_ids))
            entity2_id = random.choice(list(environment.used_ids))
            environment.add_interaction(entity1_id, entity2_id, interaction)

    # Visualize the graph
    environment.visualize()
