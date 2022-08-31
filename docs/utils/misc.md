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

[Show source in misc.py:13](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L13)

#### Signature

```python
class BlackHole(object):
    ...
```



## CheckpointManager

[Show source in misc.py:22](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L22)

#### Signature

```python
class CheckpointManager(object):
    def __init__(self, save_dir, logger=BlackHole()):
        ...
```

### CheckpointManager().get_best_ckpt_idx

[Show source in misc.py:51](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L51)

#### Signature

```python
def get_best_ckpt_idx(self):
    ...
```

### CheckpointManager().get_latest_ckpt_idx

[Show source in misc.py:60](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L60)

#### Signature

```python
def get_latest_ckpt_idx(self):
    ...
```

### CheckpointManager().get_worst_ckpt_idx

[Show source in misc.py:42](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L42)

#### Signature

```python
def get_worst_ckpt_idx(self):
    ...
```

### CheckpointManager().load_best

[Show source in misc.py:90](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L90)

#### Signature

```python
def load_best(self):
    ...
```

### CheckpointManager().load_latest

[Show source in misc.py:97](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L97)

#### Signature

```python
def load_latest(self):
    ...
```

### CheckpointManager().load_selected

[Show source in misc.py:104](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L104)

#### Signature

```python
def load_selected(self, file):
    ...
```

### CheckpointManager().save

[Show source in misc.py:69](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L69)

#### Signature

```python
def save(self, model, args, score, others=None, step=None):
    ...
```



## get_logger

[Show source in misc.py:115](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L115)

#### Signature

```python
def get_logger(name, log_dir=None):
    ...
```



## get_new_log_dir

[Show source in misc.py:134](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L134)

#### Signature

```python
def get_new_log_dir(root="./logs", postfix="", prefix=""):
    ...
```



## int_list

[Show source in misc.py:148](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L148)

#### Signature

```python
def int_list(argstr):
    ...
```



## int_tuple

[Show source in misc.py:140](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L140)

#### Signature

```python
def int_tuple(argstr):
    ...
```



## log_hyperparams

[Show source in misc.py:156](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L156)

#### Signature

```python
def log_hyperparams(writer, args):
    ...
```



## seed_all

[Show source in misc.py:109](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L109)

#### Signature

```python
def seed_all(seed):
    ...
```



## str_list

[Show source in misc.py:152](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L152)

#### Signature

```python
def str_list(argstr):
    ...
```



## str_tuple

[Show source in misc.py:144](https://github.com/enricobu96/myMID/blob/main/utils/misc.py#L144)

#### Signature

```python
def str_tuple(argstr):
    ...
```


