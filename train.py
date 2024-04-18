import os
import random

import torch
import wandb
from torch import nn
from torch.optim import Adam
from torch_geometric.data import DataLoader, Data

from src.attribute_enums import Size, Classification, Diet, Relationships
from src.environments import EnvironmentGraph
from src.graph_networks import GCN, prepare_data, train
from utils.general_utils import safe_load_yaml

# Create and train the GCN model
if __name__ == "__main__":

    hyperparam_defaults = safe_load_yaml(os.path.join('configs', 'network_config.yaml'))
    data_gen_defaults = safe_load_yaml(os.path.join('configs', 'data_config.yaml'))

    all_defaults = hyperparam_defaults | data_gen_defaults
    wandb.init(project="ecosystem_graphs",
               config=all_defaults)
    config = wandb.config

    environment = EnvironmentGraph()

    # Add life forms to the graph with attributes
    for i in range(config.num_lifeforms):
        position = (random.uniform(config.x_position['min'], config.x_position['max']), random.uniform(config.y_position['min'], config.y_position['max']))  # Random position within a 10x10 grid
        speed = random.uniform(config.speed['min'], config.speed['max'])
        size = random.choice(list(Size))
        diet = random.choice(list(Diet))
        classification = random.choice(list(Classification)[:3])
        environment.add_life_form(position, speed, size, diet, classification)

    # Add resources to the graph with attributes
    for i in range(config.num_resources):
        position = (random.uniform(config.x_position['min'], config.x_position['max']), random.uniform(config.y_position['min'], config.y_position['max']))  # Random position within a 10x10 grid
        availability = random.uniform(config.resource_availability['min'], config.resource_availability['max'])
        environment.add_resource(position, availability)

    # Add random interactions between entities
    interactions = [(Relationships.CONSUMES, config.num_interactions['consumes']), (Relationships.PREDATORPREY, config.num_interactions['predatorprey'])]

    for interaction, num in interactions:
        for _ in range(num):
            entity1_id = random.choice(list(environment.used_ids))
            entity2_id = random.choice(list(environment.used_ids))
            environment.add_interaction(entity1_id, entity2_id, interaction)

    if config.num_lifeforms+config.num_resources <= 100:
        environment.visualize()

    # Prepare data
    train_data = prepare_data(environment)

    # Instantiate the model
    model = GCN(num_node_features=train_data.num_features, hidden_channels=64)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=config.lr)

    # Train the model
    train(model, train_data, optimizer, criterion, epochs=config.epochs)
