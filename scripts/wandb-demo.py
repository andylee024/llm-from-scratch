
import random
import os
import wandb

from dotenv import load_dotenv
load_dotenv()

run = wandb.init(
    # Get the wandb entity from environment variables
    entity=os.getenv("WANDB_ENTITY"),
    # Get the wandb project from environment variables
    project=os.getenv("WANDB_PROJECT"),
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()