#  Copyright (c) 2024. Kromek Group Ltd.
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.environments import EnvironmentGraph

from src.attribute_enums import Size, Classification, Diet, Species, Relationships


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc3 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = self.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        return x



class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x


def prepare_data(environment):
    # Extract node features and adjacency matrix from EnvironmentGraph
    node_features = []
    adjacency_matrices = []
    labels = []
    id_to_index = {node_id: i for i, node_id in enumerate(environment.used_ids)}

    max_feature_length = max(len(node_data) for node_data in environment.graph.nodes.values())

    for node_id, node_data in environment.graph.nodes(data=True):
        if node_data['type'] == 'life_form':
            position = node_data['position']
            features = [position[0], position[1], node_data['classification'].value, node_data['speed'], node_data['size'].value, node_data['diet'].value, node_data['species'].value]
        elif node_data['type'] == 'resource':
            position = node_data['position']
            features = [position[0], position[1], node_data['availability']]
        else:
            features = []  # Handle other types of nodes

        # Pad features with zeros to ensure consistent length
        padded_features = features + [0] * (max_feature_length - len(features))
        node_features.append(padded_features)

    node_features = np.array(node_features)  # Convert to NumPy array

    for edge in environment.graph.edges(data=True):
        source_id, target_id, data = edge
        adjacency_matrix = np.zeros((len(environment.graph.nodes()), len(environment.graph.nodes())))
        source_index = id_to_index[source_id]
        target_index = id_to_index[target_id]
        adjacency_matrix[source_index][target_index] = 1
        adjacency_matrices.append(adjacency_matrix)
        labels.append(data['relationship'].value)

    node_features = torch.tensor(node_features, dtype=torch.float32)
    adjacency_matrices = torch.tensor(adjacency_matrices, dtype=torch.float32)
    labels =  torch.nn.functional.one_hot(torch.tensor(labels), len(Relationships))

    return node_features, adjacency_matrices, labels


def train_model(model, node_features, adjacency_matrices, labels, epochs=100, lr=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(node_features, adjacency_matrices)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# Create and train the GCN model
if __name__ == "__main__":
    environment = EnvironmentGraph()


    # Add 4 life forms to the graph with attributes
    for i in range(1, 10):
        position = (random.uniform(0, 10), random.uniform(0, 10))  # Random position within a 10x10 grid
        speed = random.uniform(0, 10)
        size = random.choice(list(Size))
        diet = random.choice(list(Diet))
        classification = random.choice(list(Classification)[:3])
        environment.add_life_form(position, speed, size, diet, classification)

    # Add 4 resources to the graph with attributes
    for i in range(1, 10):
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

    environment.visualize()
    # Prepare data
    node_features, adjacency_matrices, labels = prepare_data(environment)

    # Define model and train
    input_dim = 7  # Define input dimension based on node features
    hidden_dim = 64  # Define hidden dimension
    output_dim = 2  # Define output dimension (binary classification)
    model = GCN(input_dim, hidden_dim, output_dim)
    train_model(model, node_features, adjacency_matrices, labels)
    print('hello')