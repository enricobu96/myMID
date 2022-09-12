# Data

[Mymid Index](../README.md#mymid-index) /
[Utils](./index.md#utils) /
Data

> Auto-generated documentation for [utils.data](https://github.com/enricobu96/myMID/blob/main/utils/data.py) module.

- [Data](#data)
  - [get_data_iterator](#get_data_iterator)
  - [get_train_val_test_datasets](#get_train_val_test_datasets)
  - [get_train_val_test_loaders](#get_train_val_test_loaders)

## get_data_iterator

[Show source in data.py:43](https://github.com/enricobu96/myMID/blob/main/utils/data.py#L43)

Allows training with DataLoaders in a single infinite loop:
for i, data in enumerate(inf_generator(train_loader)):

#### Signature

```python
def get_data_iterator(iterable):
    ...
```



## get_train_val_test_datasets

[Show source in data.py:17](https://github.com/enricobu96/myMID/blob/main/utils/data.py#L17)

Takes in input dataset and train-validation ratio, returns a tuple of datasets: training, validation and test.

#### Signature

```python
def get_train_val_test_datasets(dataset, train_ratio, val_ratio):
    ...
```



## get_train_val_test_loaders

[Show source in data.py:30](https://github.com/enricobu96/myMID/blob/main/utils/data.py#L30)

#### Signature

```python
def get_train_val_test_loaders(
    dataset, train_ratio, val_ratio, train_batch_size, val_test_batch_size, num_workers
):
    ...
```


