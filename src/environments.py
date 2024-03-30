import random
from enum import Enum

import networkx as nx
from matplotlib import pyplot as plt

from src.attribute_enums import Size, Classification, Species, Diet, Relationships

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
        if classification == Classification.ANIMAL:
            species = random.choice(list(Species)[0:5])  # Select from animal species
        elif classification == Classification.INSECT:
            species = random.choice(list(Species)[5:10])  # Select from insect species
        elif classification == Classification.PLANT:
            species = random.choice(list(Species)[10:15])  # Select from plant species
        else:
            raise ValueError("Invalid classification. Must be one of 'ANIMAL', 'INSECT', or 'PLANT'.")

        life_form_id = self._generate_id(species.name)
        if life_form_id in self.used_ids:
            raise ValueError(f"ID {life_form_id} is already in use.")

        self.graph.add_node(life_form_id, type='life_form', position=position,
                            speed=speed, size=size, diet=diet,
                            classification=classification, species=species)
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
        if self.graph.nodes[entity1_id]['type'] == 'life_form':
            entity1_class = self.graph.nodes[entity1_id]['classification']
        else:
            entity1_class = Classification.RESOURCE
        if self.graph.nodes[entity2_id]['type'] == 'life_form':
            entity2_class = self.graph.nodes[entity2_id]['classification']
        else:
            entity2_class = Classification.RESOURCE
        if relationship == Relationships.PREDATORPREY:
            if not (entity1_class in [Classification.ANIMAL, Classification.INSECT]) or entity2_class == Classification.RESOURCE:
                return  # Only allow predator-prey interactions between animals/insects and resources
        elif relationship == Relationships.CONSUMES:
            if not ((entity1_class in [Classification.ANIMAL, Classification.INSECT, Classification.PLANT]) and entity2_class == Classification.RESOURCE) \
                    and not ((entity1_class in [Classification.ANIMAL, Classification.INSECT]) and entity2_class == Classification.PLANT):
                return  # Only allow consumes interactions between life forms and resources
        self.graph.add_edge(entity1_id, entity2_id, relationship=relationship)

    def update(self):
        # Update the graph dynamically (if needed)
        pass

    def visualize(self):
        pos = nx.spring_layout(self.graph, k=5)  # Define layout for visualization and adjust spacing
        node_colors = []
        edge_labels = {}

        for node_id, node_data in self.graph.nodes(data=True):
            if node_data['type'] == 'resource':
                node_colors.append('brown')
            elif node_data['classification'] == Classification.PLANT:
                node_colors.append('lightgreen')
            elif node_data['classification'] == Classification.INSECT:
                node_colors.append('blue')
            elif node_data['classification'] == Classification.ANIMAL:
                node_colors.append('red')

        for edge in self.graph.edges(data=True):
            source, target, data = edge
            relationship = data.get('relationship', '')
            edge_labels[(source, target)] = relationship.name

        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, arrows=True)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()

