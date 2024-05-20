FROM 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-ec2

RUN pip install \
    lightning \
    metaflow \
    wandb \
    tiktoken
