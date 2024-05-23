# Resampling-free Particle Filters in High-dimensions

## Overview
This repository contains code for ["Resampling-free Particle Filters in High-dimensions"](https://arxiv.org/pdf/2404.13698). In this work, we develop a new particle filter algorithm designed for inference in high-dimensional spaces. Unlike many prior particle filter algorithms, our approach avoids particle resampling. This helps it avoid particle deprivation, an issue in which density in large regions of the state space may be lost due to particles not being resampled in those areas. The proposed algorithm is effective in a high-dimensional synthetic localization task and a 6D pose estimation task.

## Prerequisites
This repository requires the following Python libraries: Numpy and Matplotlib.

To set up the necessary environment, please follow these steps:

1. Clone this repository:
```
git clone https://github.com/FieteLab/Wide-Network-Alignment
cd Wide-Network-Alignment
```

2. Install required packages.
```
pip install numpy
pip install matplotlib
```

## How to Run

Please run the following code to compare our algorithm with Monte Carlo Localization (MCL) on a synthetic localization task.
```
python3 localization.py
```
