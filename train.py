#  Copyright (c) 2024. Kromek Group Ltd.
import random

import mlflow.pytorch

from src.environments import EnvironmentGraph

from src.attribute_enums import Size, Classification, Diet, Relationships
from src.graph_networks import GCN, prepare_data, train_model

# Create and train the GCN model
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://192.168.1.94:8080")  # Set your MLflow tracking URI
    mlflow.set_experiment("ecosystem_graphs")  # Set your MLflow experiment name

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

    import wandb
    # Define model and train
    hyperparam_defaults = dict(
        input_dim = 7,  # Define input dimension based on node features
        hidden_dim = 64,  # Define hidden dimension
        output_dim = 2,  # Define output dimension (binary classification)
        epochs = 100,
        lr = 0.1,
    )
    wandb.init(project="ecosystem_graphs",
               config=hyperparam_defaults)
    config = wandb.config
    model = GCN(config.input_dim, config.hidden_dim, config.output_dim)
    wandb.watch(model)
    train_model(model, node_features, adjacency_matrices, labels, epochs=config.epochs, lr=config.lr)

    mlflow.pytorch.log_model(model, "models")

    mlflow.end_run()