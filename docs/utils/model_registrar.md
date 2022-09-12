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

Just a utility class for building the model (not crucial for model structure):
- Sets the directory and the device
- Provide get methods for model
- Provide saving and loading utilities

#### Signature

```python
class ModelRegistrar(nn.Module):
    def __init__(self, model_dir, device):
        ...
```

### ModelRegistrar().forward

[Show source in model_registrar.py:23](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L23)

#### Signature

```python
def forward(self):
    ...
```

### ModelRegistrar().get_all_but_name_match

[Show source in model_registrar.py:49](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L49)

#### Signature

```python
def get_all_but_name_match(self, name):
    ...
```

### ModelRegistrar().get_model

[Show source in model_registrar.py:26](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L26)

#### Signature

```python
def get_model(self, name, model_if_absent=None):
    ...
```

### ModelRegistrar().get_name_match

[Show source in model_registrar.py:42](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L42)

#### Signature

```python
def get_name_match(self, name):
    ...
```

### ModelRegistrar().load_models

[Show source in model_registrar.py:66](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L66)

#### Signature

```python
def load_models(self, model_dict):
    ...
```

### ModelRegistrar().print_model_names

[Show source in model_registrar.py:56](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L56)

#### Signature

```python
def print_model_names(self):
    ...
```

### ModelRegistrar().save_models

[Show source in model_registrar.py:59](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L59)

#### Signature

```python
def save_models(self, save_path):
    ...
```

### ModelRegistrar().to

[Show source in model_registrar.py:77](https://github.com/enricobu96/myMID/blob/main/utils/model_registrar.py#L77)

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


