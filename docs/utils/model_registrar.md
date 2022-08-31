# ModelRegistrar

[Mymid Index](../README.md#mymid-index) /
[Utils](./index.md#utils) /
ModelRegistrar

> Auto-generated documentation for [utils.model_registrar](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py) module.

- [ModelRegistrar](#modelregistrar)
  - [ModelRegistrar](#modelregistrar-1)
    - [ModelRegistrar().forward](#modelregistrar()forward)
    - [ModelRegistrar().get_all_but_name_match](#modelregistrar()get_all_but_name_match)
    - [ModelRegistrar().get_model](#modelregistrar()get_model)
    - [ModelRegistrar().get_name_match](#modelregistrar()get_name_match)
    - [ModelRegistrar().load_models](#modelregistrar()load_models)
    - [ModelRegistrar().print_model_names](#modelregistrar()print_model_names)
    - [ModelRegistrar().save_models](#modelregistrar()save_models)
    - [ModelRegistrar().to](#modelregistrar()to)
  - [get_model_device](#get_model_device)

## ModelRegistrar

[Show source in model_registrar.py:10](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L10)

#### Signature

```python
class ModelRegistrar(nn.Module):
    def __init__(self, model_dir, device):
        ...
```

### ModelRegistrar().forward

[Show source in model_registrar.py:17](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L17)

#### Signature

```python
def forward(self):
    ...
```

### ModelRegistrar().get_all_but_name_match

[Show source in model_registrar.py:43](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L43)

#### Signature

```python
def get_all_but_name_match(self, name):
    ...
```

### ModelRegistrar().get_model

[Show source in model_registrar.py:20](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L20)

#### Signature

```python
def get_model(self, name, model_if_absent=None):
    ...
```

### ModelRegistrar().get_name_match

[Show source in model_registrar.py:36](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L36)

#### Signature

```python
def get_name_match(self, name):
    ...
```

### ModelRegistrar().load_models

[Show source in model_registrar.py:60](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L60)

#### Signature

```python
def load_models(self, model_dict):
    ...
```

### ModelRegistrar().print_model_names

[Show source in model_registrar.py:50](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L50)

#### Signature

```python
def print_model_names(self):
    ...
```

### ModelRegistrar().save_models

[Show source in model_registrar.py:53](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L53)

#### Signature

```python
def save_models(self, save_path):
    ...
```

### ModelRegistrar().to

[Show source in model_registrar.py:71](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L71)

#### Signature

```python
def to(self, device):
    ...
```



## get_model_device

[Show source in model_registrar.py:6](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L6)

#### Signature

```python
def get_model_device(model):
    ...
```


