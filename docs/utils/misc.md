# Misc

[Mymid Index](../README.md#mymid-index) /
[Utils](./index.md#utils) /
Misc

> Auto-generated documentation for [utils.misc](https://github.com/enricobu96/myMID/blob/main/utils/misc.py) module.

- [Misc](#misc)
  - [BlackHole](#blackhole)
  - [CheckpointManager](#checkpointmanager)
    - [CheckpointManager().get_best_ckpt_idx](#checkpointmanager()get_best_ckpt_idx)
    - [CheckpointManager().get_latest_ckpt_idx](#checkpointmanager()get_latest_ckpt_idx)
    - [CheckpointManager().get_worst_ckpt_idx](#checkpointmanager()get_worst_ckpt_idx)
    - [CheckpointManager().load_best](#checkpointmanager()load_best)
    - [CheckpointManager().load_latest](#checkpointmanager()load_latest)
    - [CheckpointManager().load_selected](#checkpointmanager()load_selected)
    - [CheckpointManager().save](#checkpointmanager()save)
  - [get_logger](#get_logger)
  - [get_new_log_dir](#get_new_log_dir)
  - [int_list](#int_list)
  - [int_tuple](#int_tuple)
  - [log_hyperparams](#log_hyperparams)
  - [seed_all](#seed_all)
  - [str_list](#str_list)
  - [str_tuple](#str_tuple)

## BlackHole

[Show source in misc.py:17](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L17)

#### Signature

```python
class BlackHole(object):
    ...
```



## CheckpointManager

[Show source in misc.py:26](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L26)

#### Signature

```python
class CheckpointManager(object):
    def __init__(self, save_dir, logger=BlackHole()):
        ...
```

### CheckpointManager().get_best_ckpt_idx

[Show source in misc.py:55](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L55)

#### Signature

```python
def get_best_ckpt_idx(self):
    ...
```

### CheckpointManager().get_latest_ckpt_idx

[Show source in misc.py:64](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L64)

#### Signature

```python
def get_latest_ckpt_idx(self):
    ...
```

### CheckpointManager().get_worst_ckpt_idx

[Show source in misc.py:46](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L46)

#### Signature

```python
def get_worst_ckpt_idx(self):
    ...
```

### CheckpointManager().load_best

[Show source in misc.py:94](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L94)

#### Signature

```python
def load_best(self):
    ...
```

### CheckpointManager().load_latest

[Show source in misc.py:101](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L101)

#### Signature

```python
def load_latest(self):
    ...
```

### CheckpointManager().load_selected

[Show source in misc.py:108](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L108)

#### Signature

```python
def load_selected(self, file):
    ...
```

### CheckpointManager().save

[Show source in misc.py:73](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L73)

#### Signature

```python
def save(self, model, args, score, others=None, step=None):
    ...
```



## get_logger

[Show source in misc.py:119](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L119)

#### Signature

```python
def get_logger(name, log_dir=None):
    ...
```



## get_new_log_dir

[Show source in misc.py:138](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L138)

#### Signature

```python
def get_new_log_dir(root="./logs", postfix="", prefix=""):
    ...
```



## int_list

[Show source in misc.py:152](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L152)

#### Signature

```python
def int_list(argstr):
    ...
```



## int_tuple

[Show source in misc.py:144](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L144)

#### Signature

```python
def int_tuple(argstr):
    ...
```



## log_hyperparams

[Show source in misc.py:160](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L160)

#### Signature

```python
def log_hyperparams(writer, args):
    ...
```



## seed_all

[Show source in misc.py:113](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L113)

#### Signature

```python
def seed_all(seed):
    ...
```



## str_list

[Show source in misc.py:156](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L156)

#### Signature

```python
def str_list(argstr):
    ...
```



## str_tuple

[Show source in misc.py:148](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L148)

#### Signature

```python
def str_tuple(argstr):
    ...
```


