# Data Structures

[Mymid Index](../README.md#mymid-index) /
[Environment](./index.md#environment) /
Data Structures

> Auto-generated documentation for [environment.data_structures](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py) module.

- [Data Structures](#data-structures)
  - [DoubleHeaderNumpyArray](#doubleheadernumpyarray)
    - [DoubleHeaderNumpyArray().get_single_header_array](#doubleheadernumpyarray()get_single_header_array)
  - [RingBuffer](#ringbuffer)
    - [RingBuffer().append](#ringbuffer()append)
    - [RingBuffer().appendleft](#ringbuffer()appendleft)
    - [RingBuffer().dtype](#ringbuffer()dtype)
    - [RingBuffer().extend](#ringbuffer()extend)
    - [RingBuffer().extendleft](#ringbuffer()extendleft)
    - [RingBuffer().is_full](#ringbuffer()is_full)
    - [RingBuffer().maxlen](#ringbuffer()maxlen)
    - [RingBuffer().pop](#ringbuffer()pop)
    - [RingBuffer().popleft](#ringbuffer()popleft)
    - [RingBuffer().shape](#ringbuffer()shape)
  - [SingleHeaderNumpyArray](#singleheadernumpyarray)

## DoubleHeaderNumpyArray

[Show source in data_structures.py:183](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L183)

#### Signature

```python
class DoubleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        ...
```

### DoubleHeaderNumpyArray().get_single_header_array

[Show source in data_structures.py:211](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L211)

#### Signature

```python
def get_single_header_array(self, h1: str, rows=slice(None, None, None)):
    ...
```



## RingBuffer

[Show source in data_structures.py:6](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L6)

#### Signature

```python
class RingBuffer(Sequence):
    def __init__(self, capacity, dtype=float, allow_overwrite=True):
        ...
```

### RingBuffer().append

[Show source in data_structures.py:69](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L69)

#### Signature

```python
def append(self, value):
    ...
```

### RingBuffer().appendleft

[Show source in data_structures.py:82](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L82)

#### Signature

```python
def appendleft(self, value):
    ...
```

### RingBuffer().dtype

[Show source in data_structures.py:56](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L56)

#### Signature

```python
@property
def dtype(self):
    ...
```

### RingBuffer().extend

[Show source in data_structures.py:111](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L111)

#### Signature

```python
def extend(self, values):
    ...
```

### RingBuffer().extendleft

[Show source in data_structures.py:135](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L135)

#### Signature

```python
def extendleft(self, values):
    ...
```

### RingBuffer().is_full

[Show source in data_structures.py:47](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L47)

True if there is no more space in the buffer

#### Signature

```python
@property
def is_full(self):
    ...
```

### RingBuffer().maxlen

[Show source in data_structures.py:65](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L65)

#### Signature

```python
@property
def maxlen(self):
    ...
```

### RingBuffer().pop

[Show source in data_structures.py:95](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L95)

#### Signature

```python
def pop(self):
    ...
```

### RingBuffer().popleft

[Show source in data_structures.py:103](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L103)

#### Signature

```python
def popleft(self):
    ...
```

### RingBuffer().shape

[Show source in data_structures.py:60](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L60)

#### Signature

```python
@property
def shape(self):
    ...
```



## SingleHeaderNumpyArray

[Show source in data_structures.py:251](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L251)

#### Signature

```python
class SingleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        ...
```


