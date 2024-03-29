import random

import matplotlib.pyplot as plt
import networkx as nx


class EnvironmentGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Use a directed graph for interactions
        self.used_ids = set()
        self.id_counter = {}

    def _generate_id(self, species):
        if species not in self.id_counter:
            self.id_counter[species] = 1
        else:
            self.id_counter[species] += 1
        return f"{species}_{self.id_counter[species]}"

    def add_life_form(self, position, speed, size, diet, classification):
        life_form_id = self._generate_id(classification)
        if life_form_id in self.used_ids:
            raise ValueError(f"ID {life_form_id} is already in use.")
        self.graph.add_node(life_form_id, type='life_form', position=position,
                            speed=speed, size=size, diet=diet, classification=classification)
        self.used_ids.add(life_form_id)

    def add_resource(self, position, availability):
        resource_id = self._generate_id("Resource")
        if resource_id in self.used_ids:
            raise ValueError(f"ID {resource_id} is already in use.")
        self.graph.add_node(resource_id, type='resource', position=position, availability=availability)
        self.used_ids.add(resource_id)

    def add_interaction(self, entity1_id, entity2_id, relationship):
        if entity1_id == entity2_id:
            return  # Prevent entities from interacting with themselves
        if relationship == 'predator-prey':
            if not ('Animal' in entity1_id or 'Insect' in entity1_id) or 'Resource' in entity2_id:
                return  # Only allow predator-prey interactions between animals/insects and resources
        elif relationship == 'consumes':
            print(entity1_id, entity2_id, relationship)

            if not (('Animal' in entity1_id or 'Insect' in entity1_id or 'Plant' in entity1_id) and 'Resource' in entity2_id)\
                    and not (('Animal' in entity1_id or 'Insect' in entity1_id) and 'Plant' in entity2_id):
                return  # Only allow consumes interactions between life forms and resources
        self.graph.add_edge(entity1_id, entity2_id, relationship=relationship)

    def update(self):
        # Update the graph dynamically (if needed)
        pass

    def visualize(self):
        pos = nx.spring_layout(self.graph, k=5)  # Define layout for visualization and adjust spacing
        node_colors = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data['type'] == 'resource':
                node_colors.append('brown')
            elif node_data['classification'] == 'Plant':
                node_colors.append('lightgreen')
            elif node_data['classification'] == 'Insect':
                node_colors.append('blue')
            elif node_data['classification'] == 'Animal':
                node_colors.append('red')

        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, arrows=True)
        edge_labels = {(u, v): d["relationship"] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()


# Example usage:
if __name__ == "__main__":
    environment = EnvironmentGraph()

    # Add 4 life forms to the graph with attributes
    for i in range(1, 5):
        position = (random.uniform(0, 10), random.uniform(0, 10))  # Random position within a 10x10 grid
        speed = random.uniform(0, 10)
        size = random.choice(['Small', 'Medium', 'Large'])
        diet = random.choice(['Carnivore', 'Herbivore', 'Omnivore'])
        classification = random.choice(['Animal', 'Plant', 'Insect'])
        environment.add_life_form(position, speed, size, diet, classification)

    # Add 4 resources to the graph with attributes
    for i in range(1, 5):
        position = (random.uniform(0, 10), random.uniform(0, 10))  # Random position within a 10x10 grid
        availability = random.uniform(0, 100)
        environment.add_resource(position, availability)

    # Add random interactions between entities
    interactions = [('consumes', 6), ('consumes', 6), ('consumes', 6), ('consumes', 6),
                    ('consumes', 6), ('consumes', 6), ('consumes', 6), ('consumes', 6),
                    ('predator-prey', 6), ('predator-prey', 6), ('predator-prey', 6), ('predator-prey', 6),
                    ]

    for interaction, num in interactions:
        for _ in range(num):
            this_entity1_id = random.choice(list(environment.used_ids))
            this_entity2_id = random.choice(list(environment.used_ids))
            environment.add_interaction(this_entity1_id, this_entity2_id, interaction)

    # Visualize the graph
    environment.visualize()
