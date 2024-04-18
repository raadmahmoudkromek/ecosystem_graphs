import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(2 * hidden_channels, 1)  # Adjusted output size for edge prediction

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Gather source and target node embeddings
        source = x[edge_index[0]]
        target = x[edge_index[1]]

        # Concatenate source and target embeddings to represent edges
        edge_features = torch.cat([source, target], dim=-1)

        # Pass edge features through linear layer for prediction
        out = self.fc(edge_features).squeeze(1)
        return torch.sigmoid(out)


import torch
import numpy as np
from torch_geometric.data import Data


def prepare_data(environment):
    # Extract node features and adjacency matrix from EnvironmentGraph
    node_features = []
    edge_list = []
    labels = []
    id_to_index = {node_id: i for i, node_id in enumerate(environment.used_ids)}

    for node_id, node_data in environment.graph.nodes(data=True):
        if node_data['type'] == 'life_form':
            position = node_data['position']
            features = [position[0], position[1], node_data['classification'].value, node_data['speed'],
                        node_data['size'].value, node_data['diet'].value, node_data['species'].value, 0]
        elif node_data['type'] == 'resource':
            position = node_data['position']
            features = [position[0], position[1], 0, 0, 0, 0, 0, node_data['availability']]
        else:
            features = []  # Handle other types of nodes

        node_features.append(features)

    for edge in environment.graph.edges(data=True):
        source_id, target_id, data = edge
        source_index = id_to_index[source_id]
        target_index = id_to_index[target_id]
        edge_list.append([source_index, target_index])
        labels.append(data['relationship'].value)

    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_list = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    labels = torch.tensor(np.array(labels), dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_list, y=labels)
    return data


def train(model, dataset, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        out = model(dataset.x, dataset.edge_index)
        loss = criterion(out, dataset.y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({"loss": loss.item()})
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss}')
