# Dump gpt2 model weights to local file system using Pytorch
# Should be run from master branch

import torch
from model import GPT

# Load the model
model_name = "gpt2"
model = GPT.from_pretrained(model_name)

# Save the model
torch.save(model.state_dict(), "models/gpt2.pth")
