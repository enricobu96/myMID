# Official Implementation for CVPR2022 Paper "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"

By Tianpei Gu*, Guangyi Chen*, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou and Jiwen Lu

## Environment

We use pytorch 1.7.1 with cuda > 10.1 for all experiments.


## Prepare Data

```
python process_data.py
```

## Train & Test

First modify or create your own config file in ```/configs``` and run ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET``` where ```$DATASET``` should from ["eth", "hotel", "univ", "zara1", "zara2", "sdd"]

## Documentation

In order to get documentation for the project, install the following pip modules:

```
pip install headsdown mkdocs
```

Then generate the .md documentation files with:

```
python -m handsdown --external `git config --get remote.origin.url` --create-configs
```

And then build and serve the documentation with:

```
python -m mkdocs build
python -m mkdocs serve
```

The documentation is now available on [http://127.0.0.1/8000/](http://127.0.0.1/8000/).