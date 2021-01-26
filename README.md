# TF2_pruning_custom
Core functions for pruning weights implemented in TF2.

Pruning Strategy for TensorFlow 2.x

- vanilla weight pruning based on magnitude
- block weight pruning based on magnitude (recommended for inference speedup)

## Test example
``` 
    shape = [4, 4]
    block_size = [2, 2]
    sparse_ratio = 0.80

    weight = tf.random.normal(shape)
    print(weight)

    result = block_pruning(weight, block_size, sparse_ratio)
    print(result)

    result = vanilla_pruning(weight, sparse_ratio)
    print(result)
```
running result:
raw weight matrix:
``` 
tf.Tensor(
[[-0.43940789  0.8904683  -0.1364473  -0.41253278]
 [-0.29909924  0.42642906  1.1408455  -0.46779928]
 [ 1.2219299   0.61770624 -0.7310449   0.4641536 ]
 [ 0.12570195 -0.5342898   1.1881508  -0.7505593 ]], shape=(4, 4), dtype=float32)
 ```
 
 block sparsed:
 ```
tf.Tensor(
[[-0.         0.        -0.        -0.       ]
 [-0.         0.         0.        -0.       ]
 [ 0.         0.        -0.7310449  0.4641536]
 [ 0.        -0.         1.1881508 -0.7505593]], shape=(4, 4), dtype=float32)
 ```
 
 randomly sparsed:
 ```
tf.Tensor(
[[0.        0.        0.        0.       ]
 [0.        0.        1.1408455 0.       ]
 [1.2219299 0.        0.        0.       ]
 [0.        0.        1.1881508 0.       ]], shape=(4, 4), dtype=float32)
```
