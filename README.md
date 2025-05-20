# cloud-slurm
My personal Slurm cluster to orchestrate compute nodes across different GPU providers for multi-node multi-gpu model training; effectively creating a mesh network through Tailscale.

Don't expect this to always be up to date but you might be able to use it as some sort of template for your own. You don't have to use the same GPU providers; so long as you can SSH into a VM instance and/or deploy a Docker container into them.

## Table of Contents
### Setup
1. [Prerequisites](#main-prerequisites)
2. [Tailscale Setup](#tailscale-setup)
   - [Generating an Auth Key](#generating-an-auth-key)
3. [Head Node Setup](#head-node-setup)
   - [Munge](#munge)
   - [Slurm](#slurm)
4. [Compute Node Setup](#compute-node-setup)
   - [Paperspace](#paperspace)
   - [Lambda Labs](#lambda-labs)
   - [Runpods](#runpods)
5. [Network Storage Setup](#network-storage-setup)

### Usage & Training
1. [Using Slurm](#using-slurm)
2. [Profiling](#profiling)
3. [Using PyTorch DDP with Slurm](#using-pytorch-ddp-with-slurm)
4. [Training On A Toy Example](#training-on-a-toy-example)
5. [Full Training](#full-training)



## Main Prerequisites
1. [Tailscale](tailscale.com) -  Private VPN for your head and compute nodes to be within.
2. [Paperspace](paperspace.com)- Cloud GPU Provider
3. [Runpod](runpod.io) - Cloud GPU Provider
4. [Lambda Labs](lambda.ai) - Cloud GPU Provider
5. [NVIDIA NSight Systems](https://developer.nvidia.com/nsight-systems/get-started) - GPU Profiling
6. [MLFlow](https://mlflow.org/) - MLOps platform to track training progress.

## Tailscale Setup
Register and subscribe to a [plan](https://tailscale.com/pricing) on Tailscale. Each of our nodes need to communicate with each other through some network and the easiest way to do this is to have them all connected to a single VPN with assigned static IP addresses. (More on this later)

### Generating an Auth Key
Once you've done that we'll have to generate an Auth key so that our nodes can connect into the VPN. 
Navigate through the web platform:

`Settings > Keys (Under Personal Settings) > Generate auth key...`

This part is entirely dependent on your security concerns when setting the expiration, ephemerality, and reusability. 

I do the following:
| Setting | State |
| ------- | ----- |
| Expiration | However long I feel that it would take to finish off my training job.|
| Ephemeral | Off. Since we are operating across the cloud and across different services our nodes may not be stable enough to maintain 24/7 uptime. Especially if you are using the Community Cloud in Runpod. |
| Reusable | On. I'm kind of lazy and although this is a big risk, having one key to set across all your nodes is convenient. Otherwise, you will have to maintain multiple Auth keys for your nodes if you want.|

One thing that sucks about  any of this is that when your Auth Key becomes invalid you will have to rotate through all your nodes and set them to a new one. There may be a way to automate this or do this a better way but this is the easiest solution I fell upon.

## Head Node Setup
I have my personal desktop as both a head and compute node with an **RTX A6000**. But you can extrapolate these instructions to just an exclusive head node either locally or on the cloud. I'd recommend using a cheap non-GPU instance for the head node if that is the case.

### Munge

### Slurm

## Compute Node Setup


### Paperspace

### Lambda Labs

### Runpods

## Network Storage Setup
I use a shared network drive, but you can also just copy your datasets over to each disk drive per compute node. There's pros and cons to each; if you need read speed and want to use
**LMDB** it may be worthwhile. However, I'll be using **WebDataset** for my shared network drive. My drive is located on the Head node which is my local desktop. It is a **4TB NVMe M.2 SSD**. The faster the read/write capabilities of the drive the better. You'll still have to consider network bandwidth for each node though!

### Setting up NFS
Install the NFS Server wherever you plan on hosting this drive. For my case, it will be my head node.

#### Head Node
```
sudo apt update
sudo apt install nfs-kernel-server
sudo mkdir -p /mnt/shared-slurm
sudo chown $USER:$USER /mnt/shared-slurm
```

Add all your tailscale node IPs to this file: `/etc/exports`. The lines should like this. To display the current device's IP address run `tailscale ip`.
```
/mnt/shared-slurm <TAILSCALE_IP0>/10(rw,sync,no_subtree_check,no_root_squash)
...
/mnt/shared-slurm <TAILSCALE_IPN>/10(rw,sync,no_subtree_check,no_root_squash)
```

Then refresh and restart your `nfs-kernel-server`. If you don't have systemctl in a docker container you may have to use `supervisord`.
```
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

#### Compute Node
```
sudo apt update
sudo apt install nfs-common
sudo mkdir -p /mnt/shared-slurm
sudo mount <TAILSCALE_SHARED_DRIVE_IP>:/mnt/shared-slurm /mnt/shared-slurm
```

To set up auto-mount on reboot modify the `/etc/fstab` per node and add the following line.
```
<TAILSCALE_SHARED_DRIVE_IP>:/mnt/shared-slurm /mnt/shared-slurm nfs defaults 0 0
sudo mount -a
```




## Using Slurm

## Profiling

## Using PyTorch DDP with Slurm

## Training On A Toy Example

## Full Training