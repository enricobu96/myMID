# Evaluation

[Mymid Index](../README.md#mymid-index) /
[Evaluation](./index.md#evaluation) /
Evaluation

> Auto-generated documentation for [evaluation.evaluation](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py) module.

- [Evaluation](#evaluation)
  - [batch_pcmd](#batch_pcmd)
  - [compute_ade](#compute_ade)
  - [compute_batch_statistics](#compute_batch_statistics)
  - [compute_fde](#compute_fde)
  - [compute_kde_nll](#compute_kde_nll)
  - [compute_obs_violations](#compute_obs_violations)
  - [print_batch_errors](#print_batch_errors)

## batch_pcmd

[Show source in evaluation.py:186](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L186)

#### Signature

```python
def batch_pcmd(
    prediction_output_dict,
    dt,
    max_hl,
    ph,
    node_type_enum,
    kde=True,
    obs=False,
    map=None,
    prune_ph_to_future=False,
    best_of=False,
):
    ...
```



## compute_ade

[Show source in evaluation.py:50](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L50)

#### Signature

```python
def compute_ade(predicted_trajs, gt_traj):
    ...
```



## compute_batch_statistics

[Show source in evaluation.py:98](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L98)

#### Signature

```python
def compute_batch_statistics(
    prediction_output_dict,
    dt,
    max_hl,
    ph,
    node_type_enum,
    kde=True,
    obs=False,
    map=None,
    prune_ph_to_future=False,
    best_of=False,
):
    ...
```



## compute_fde

[Show source in evaluation.py:57](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L57)

#### Signature

```python
def compute_fde(predicted_trajs, gt_traj):
    ...
```



## compute_kde_nll

[Show source in evaluation.py:62](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L62)

#### Signature

```python
def compute_kde_nll(predicted_trajs, gt_traj):
    ...
```



## compute_obs_violations

[Show source in evaluation.py:80](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L80)

#### Signature

```python
def compute_obs_violations(predicted_trajs, map):
    ...
```



## print_batch_errors

[Show source in evaluation.py:174](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L174)

#### Signature

```python
def print_batch_errors(batch_errors_list, namespace, curr_iter):
    ...
```


