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

[Show source in data_structures.py:182](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L182)

#### Signature

```python
class DoubleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        ...
```

### DoubleHeaderNumpyArray().get_single_header_array

[Show source in data_structures.py:210](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L210)

#### Signature

```python
def get_single_header_array(self, h1: str, rows=slice(None, None, None)):
    ...
```



## RingBuffer

[Show source in data_structures.py:7](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L7)

Create a new ring buffer with the given capacity and element type.
Code copy-pasted from: https://github.com/eric-wieser/numpy_ringbuffer

#### Arguments

- `capacity` - int : The maximum capacity of the ring buffer
- `dtype` - data-type, optional : Desired type of buffer elements. Use a type like (float, 2) to produce a buffer with shape (N, 2)
- `allow_overwrite` - bool : If false, throw an IndexError when trying to append to an already full buffer

#### Signature

```python
class RingBuffer(Sequence):
    def __init__(self, capacity, dtype=float, allow_overwrite=True):
        ...
```

### RingBuffer().append

[Show source in data_structures.py:68](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L68)

#### Signature

```python
def append(self, value):
    ...
```

### RingBuffer().appendleft

[Show source in data_structures.py:81](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L81)

#### Signature

```python
def appendleft(self, value):
    ...
```

### RingBuffer().dtype

[Show source in data_structures.py:55](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L55)

#### Signature

```python
@property
def dtype(self):
    ...
```

### RingBuffer().extend

[Show source in data_structures.py:110](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L110)

#### Signature

```python
def extend(self, values):
    ...
```

### RingBuffer().extendleft

[Show source in data_structures.py:134](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L134)

#### Signature

```python
def extendleft(self, values):
    ...
```

### RingBuffer().is_full

[Show source in data_structures.py:44](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L44)

True if there is no more space in the buffer

#### Signature

```python
@property
def is_full(self):
    ...
```

### RingBuffer().maxlen

[Show source in data_structures.py:64](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L64)

#### Signature

```python
@property
def maxlen(self):
    ...
```

### RingBuffer().pop

[Show source in data_structures.py:94](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L94)

#### Signature

```python
def pop(self):
    ...
```

### RingBuffer().popleft

[Show source in data_structures.py:102](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L102)

#### Signature

```python
def popleft(self):
    ...
```

### RingBuffer().shape

[Show source in data_structures.py:59](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L59)

#### Signature

```python
@property
def shape(self):
    ...
```



## SingleHeaderNumpyArray

[Show source in data_structures.py:250](https://github.com/enricobu96/myMID/blob/main/environment/data_structures.py#L250)

#### Signature

```python
class SingleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        ...
```


