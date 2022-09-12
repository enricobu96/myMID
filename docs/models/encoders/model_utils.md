# Model Utils

[Mymid Index](../../README.md#mymid-index) /
[Models](../index.md#models) /
[Encoders](./index.md#encoders) /
Model Utils

> Auto-generated documentation for [models.encoders.model_utils](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py) module.

- [Model Utils](#model-utils)
  - [CustomLR](#customlr)
    - [CustomLR().get_lr](#customlr()get_lr)
  - [ModeKeys](#modekeys)
  - [cyclical_lr](#cyclical_lr)
  - [exp_anneal](#exp_anneal)
  - [extract_subtensor_per_batch_element](#extract_subtensor_per_batch_element)
  - [mutual_inf_mc](#mutual_inf_mc)
  - [rgetattr](#rgetattr)
  - [rsetattr](#rsetattr)
  - [run_lstm_on_variable_length_seqs](#run_lstm_on_variable_length_seqs)
  - [sigmoid_anneal](#sigmoid_anneal)
  - [to_one_hot](#to_one_hot)
  - [unpack_RNN_state](#unpack_rnn_state)

## CustomLR

[Show source in model_utils.py:49](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L49)

#### Signature

```python
class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        ...
```

### CustomLR().get_lr

[Show source in model_utils.py:53](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L53)

#### Signature

```python
def get_lr(self):
    ...
```



## ModeKeys

[Show source in model_utils.py:9](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L9)

#### Signature

```python
class ModeKeys(Enum):
    ...
```



## cyclical_lr

[Show source in model_utils.py:15](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L15)

#### Signature

```python
def cyclical_lr(stepsize, min_lr=0.0003, max_lr=0.003, decay=1.0):
    ...
```



## exp_anneal

[Show source in model_utils.py:32](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L32)

#### Signature

```python
def exp_anneal(anneal_kws):
    ...
```



## extract_subtensor_per_batch_element

[Show source in model_utils.py:89](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L89)

#### Signature

```python
def extract_subtensor_per_batch_element(tensor, indices):
    ...
```



## mutual_inf_mc

[Show source in model_utils.py:58](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L58)

#### Signature

```python
def mutual_inf_mc(x_dist):
    ...
```



## rgetattr

[Show source in model_utils.py:122](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L122)

#### Signature

```python
def rgetattr(obj, attr, *args):
    ...
```



## rsetattr

[Show source in model_utils.py:115](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L115)

#### Signature

```python
def rsetattr(obj, attr, val):
    ...
```



## run_lstm_on_variable_length_seqs

[Show source in model_utils.py:64](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L64)

#### Signature

```python
def run_lstm_on_variable_length_seqs(
    lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None
):
    ...
```



## sigmoid_anneal

[Show source in model_utils.py:40](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L40)

#### Signature

```python
def sigmoid_anneal(anneal_kws):
    ...
```



## to_one_hot

[Show source in model_utils.py:28](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L28)

#### Signature

```python
def to_one_hot(labels, n_labels):
    ...
```



## unpack_RNN_state

[Show source in model_utils.py:104](https://github.com/enricobu96/myMID/blob/main/models/encoders/model_utils.py#L104)

#### Signature

```python
def unpack_RNN_state(state_tuple):
    ...
```


