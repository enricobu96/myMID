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

[Show source in evaluation.py:147](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L147)

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

[Show source in evaluation.py:11](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L11)

#### Signature

```python
def compute_ade(predicted_trajs, gt_traj):
    ...
```



## compute_batch_statistics

[Show source in evaluation.py:59](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L59)

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

[Show source in evaluation.py:18](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L18)

#### Signature

```python
def compute_fde(predicted_trajs, gt_traj):
    ...
```



## compute_kde_nll

[Show source in evaluation.py:23](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L23)

#### Signature

```python
def compute_kde_nll(predicted_trajs, gt_traj):
    ...
```



## compute_obs_violations

[Show source in evaluation.py:41](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L41)

#### Signature

```python
def compute_obs_violations(predicted_trajs, map):
    ...
```



## print_batch_errors

[Show source in evaluation.py:135](https://github.com/enricobu96/myMID/blob/main/evaluation/evaluation.py#L135)

#### Signature

```python
def print_batch_errors(batch_errors_list, namespace, curr_iter):
    ...
```


