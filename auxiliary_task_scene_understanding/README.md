# Auxiliary Task Scene Understanding

## Installation

Our code is based on [LibMTL](https://github.com/median-research-group/LibMTL). For convenience, we copy the code of LibMTL into this repository. 
And there are some minor modifications to the original code. 

It’s suggested to use **pytorch==1.10.1** and **torchvision==0.11.2** in order to reproduce our results.

After installing torch, simply run

```
pip install -r requirements.txt
```

## Dataset

Currently, only NYUv2 dataset is supported. 
To use it, you should manually download the dataset from the [Official Link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). 
You can also download it from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6d0a89f4ca1347d8af5f/?dl=1).

After downloading the dataset, you should put it under the path `data/nyuv2`. The structure of the dataset should be like

```
data/nyuv2
├── train
│   ├── depth
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   ├── ...
│   ├── ...
├── val
```

## Usage

see `pareto_atl.sh` for an example.

## Acknowledgments

We appreciate the following github repos for their valuable codebase:

- https://github.com/median-research-group/LibMTL
- https://github.com/lorenmt/mtan
