# cloud-slurm
A personal Slurm Cluster management to orchestrate compute nodes across different GPU Providers for training models. Don't expect this to always be up to date but you might be able to use it as some sort of template for your own.

## Prerequisites
1. [Tailscale](tailscale.com) -  Private VPN and mesh network for all your head and compute nodes to be within.
2. [Paperspace](paperspace.com)- Cloud GPU Provider
3. [Runpod](runpod.io) - Cloud GPU Provider
4. [Lambda Labs](lambda.ai) - Cloud GPU Provider

## Head Node Setup
I have my personal desktop as both the head and compute node. But you can extrapolate these instructions to just an exclusive head node either locally or on the cloud.

## Compute Node Setup

### Paperspace

### Lambda Labs

### Runpods

## Using Slurm

## Profiling

## Using PyTorch DDP with Slurm

## Data Management
I use a shared network drive, but you can also just copy your datasets over to each disk drive per compute node. There's pros and cons to each; if you need speed and want to use
LMDB it may be worth while. However, I'll be using WebDataset for my shared network drive. My drive is located on the Head node which is my local desktop. It is a 4TB NVMe M.2 SSD.