# PipeGCN: Efficient Full-Graph Training of Graph Convolutional Networks with Pipelined Feature Communication

Cheng Wan (Rice University), Youjie Li (UIUC), Cameron R. Wolfe (Rice University), Anastasios Kyrillidis (Rice University), Nam Sung Kim (UIUC), Yingyan Lin (Rice University)

Accepted at ICLR 2022 [[Paper](https://openreview.net/pdf?id=kSwqMH0zn1F) | [Video](https://youtu.be/kmUZIEbyypI) | [Slide](https://www.liyoujie.net/_files/ugd/b32cd5_c6bf49b80b144f5c90930fb04aff3100.pdf) | [Docker](https://hub.docker.com/r/cheng1016/pipegcn) | [Sibling](https://github.com/RICE-EIC/BNS-GCN)]


## Directory Structure

```
|-- checkpoint   # model checkpoints
|-- dataset
|-- helper       # auxiliary codes
|   `-- timer
|-- module       # PyTorch modules
|-- partitions   # partitions of input graphs
|-- results      # experiment outputs
`-- scripts      # example scripts
```

Note that `./checkpoint/`, `./dataset/`, `./partitions/` and `./results/` are empty folders at the beginning and will be created when PipeGCN is launched.

## Setup

### Environment

#### Hardware Dependencies

- A X86-CPU machine with at least 120 GB host memory 
- At least five Nvidia GPUs (at least 11 GB each)

#### Software Dependencies

- Ubuntu 18.04
- Python 3.8
- CUDA 11.1
- [PyTorch 1.8.0](https://github.com/pytorch/pytorch)
- [customized DGL 0.8.0](https://github.com/chwan-rice/dgl)
- [OGB 1.3.2](https://ogb.stanford.edu/docs/home/)

### Installation

#### Option 1: Run with Docker

We have prepared a [Docker package](https://hub.docker.com/r/cheng1016/pipegcn) for PipeGCN.

```bash
docker pull cheng1016/pipegcn
docker run --gpus all -it cheng1016/pipegcn
```

#### Option 2: Install with Conda

Running the following command will install DGL from source and other prerequisites from conda.

```bash
bash setup.sh
```

#### Option 3: Do it Yourself

Please follow the official guides ([[1]](https://github.com/pytorch/pytorch), [[2]](https://ogb.stanford.edu/docs/home/)) to install PyTorch and OGB. For DGL, please follow the [official guide](https://docs.dgl.ai/install/index.html#install-from-source) to install our customized DGL **from source** (do NOT forget to adjust the first `git clone` command to clone [our customized repo](https://github.com/chwan-rice/dgl)).  We are contacting the DGL team to integrate our modification that supports minimizing communication volume for graph partition.

### Datasets

We use Reddit, ogbn-products, Yelp and ogbn-papers100M for evaluating PipeGCN. All datasets are supposed to be stored in `./dataset/` by default. Reddit, ogbn-products and ogbn-papers100M will be downloaded by DGL or OGB automatically. Yelp is preloaded in the Docker environment, and is available [here](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) or [here](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg) (with passcode f1ao) if you choose to set up the enviromnent by yourself. 



## Basic Usage

### Core Training Options

- `--dataset`: the dataset you want to use
- `--lr`: learning rate
- `--enable-pipeline`: pipeline communication and computation
- `--feat-corr`: apply smoothing correction to stale features
- `--grad-corr`: apply smoothing correction to stale gradients
- `--corr-momentum`: the decay rate of smoothing correction
- `--n-epochs`: the number of training epochs
- `--n-partitions`: the number of partitions
- `--n-hidden`: the number of hidden units
- `--n-layers`: the number of GCN layers
- `--port`: the network port for communication
- `--no-eval`: disable evaluation process

### Run Example Scripts

Simply running `scripts/reddit.sh`, `scripts/ogbn-products.sh` and `scripts/yelp.sh` can reproduce PipeGCN under the default settings. For example, after running `bash scripts/reddit.sh`, you will get the output like this

```
...
Process 000 | Epoch 02999 | Time(s) 0.2660 | Comm(s) 0.0157 | Reduce(s) 0.0185 | Loss 0.0764
Process 001 | Epoch 02999 | Time(s) 0.2657 | Comm(s) 0.0095 | Reduce(s) 0.0683 | Loss 0.0661
Epoch 02999 | Accuracy 96.44%
model saved
Validation accuracy 96.62%
Test Result | Accuracy 97.10%
```

### Run Customized Settings

You may add/remove `--enable-pipeline`, `--feat-corr` or `--grad-corr` to reproduce the results of PipeGCN under other settings. To verify the exact throughput or time breakdown of PipeGCN, please add `--no-eval` argument to skip the evaluation step.

### Run with Multiple Compute Nodes

Our code base also supports distributed GCN training with multiple compute nodes. To achieve this, you should specify `--master-addr`, `--node-rank` and `--parts-per-node` for each compute node. An example is provided in `scripts/reddit_multi_node.sh` where we train the Reddit graph over 4 compute nodes, each of which contains 10 GPUs, with 40 partitions in total. You should run the command on each node and specify the corresponding node rank. **Please turn on `--fix-seed` argument** so that all nodes initialize the same model weights.

If the compute nodes do not share storage, you should partition the graph in a single device first and manually distribute the partitions to other compute nodes. When run the training script, please enable `--skip-partition` argument.



## Citation

```
@inproceedings{wan2022pipegcn,
  title={{PipeGCN}: Efficient Full-Graph Training of Graph Convolutional Networks with Pipelined Feature Communication},
  author={Wan, C and Li, Y and Wolfe, Cameron R and Kyrillidis, A and Kim, Nam S and Lin, Y},
  booktitle={The Tenth International Conference on Learning Representations (ICLR 2022)},
  year={2022}
}
```



## License

Copyright (c) 2022 GaTech-EIC. All rights reserved.

Licensed under the [MIT](https://github.com/RICE-EIC/PipeGCN/blob/master/LICENSE) license.
