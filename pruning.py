import numpy as np
import tensorflow as tf
# test code on TF2.4

def vanilla_pruning(weight, sparse_ratio):
    """ random pruning elements of weight matrix """

    size = tf.size(weight, out_type=tf.dtypes.float32)
    k = tf.cast(tf.round(size * sparse_ratio), tf.int32)

    reshaped_weight = tf.reshape(weight, [size])
    _, indices = tf.math.top_k(tf.math.negative(tf.math.abs(reshaped_weight)), k, sorted=True)
    indices = tf.reshape(indices, [-1, 1])

    zero_updates = tf.zeros(tf.size(indices), dtype=tf.float32)
    pruned_weight = tf.tensor_scatter_nd_update(reshaped_weight, indices, zero_updates)

    return tf.reshape(pruned_weight, tf.shape(weight))


def block_pruning(weight, block_size, sparse_ratio):
    """ (recommended) block pruning elements of weight matrix """

    num_blocks = tf.cast(tf.math.divide(tf.size(weight), block_size[0]*block_size[1]), tf.int32) 
    k = tf.cast(tf.round(tf.cast(num_blocks, tf.float32) * (1-sparse_ratio)), tf.int32)

    assert_rank_op = tf.debugging.Assert(tf.math.equal(tf.rank(weight), 2), "exception")
    with tf.control_dependencies([assert_rank_op]):
        w, h = tf.shape(weight)[-2], tf.shape(weight)[-1]

    assert_weight_op = tf.debugging.Assert(tf.math.equal(tf.math.floormod(w, block_size[0]), 0), "exception")
    assert_height_op = tf.debugging.Assert(tf.math.equal(tf.math.floormod(h, block_size[1]), 0), "exception")
    with tf.control_dependencies([assert_weight_op, assert_height_op]):
        shape = tf.shape(weight)

    count = 0
    weight_zeros = tf.zeros(tf.shape(weight))
    norm_vec = tf.TensorArray(tf.float32, size=num_blocks, clear_after_read=False)
    blocks = tf.TensorArray(tf.float32, size=num_blocks, clear_after_read=False)

    for i in tf.range(w, delta=block_size[0]):
        for j in tf.range(h, delta=block_size[1]):
            R, C = tf.meshgrid(tf.range(i, i+block_size[0]), tf.range(j, j+block_size[1]), indexing='ij')
            R_ex = tf.expand_dims(R, axis=-1)
            C_ex = tf.expand_dims(C, axis=-1)
            ind = tf.squeeze(tf.concat([R_ex, C_ex], axis=-1))
            mask = tf.tensor_scatter_nd_update(weight_zeros, ind, tf.ones(block_size))

            block = tf.slice(weight, [i, j], block_size)
            blocks = blocks.write(count, weight*mask)
            norm_ = tf.norm(block)
            norm_vec = norm_vec.write(count, norm_)
            count += 1

    norm_vec_tensor = norm_vec.stack()
    _, indices = tf.math.top_k(norm_vec_tensor, k)
    selected_blocks = blocks.gather(indices)
    result = tf.reduce_sum(selected_blocks, axis=0)

    return result
        
        
if __name__ == '__main__':
    shape = [4, 4]
    block_size = [2, 2]
    sparse_ratio = 0.80
    
    weight = tf.random.normal(shape)
    print(weight)

    result = block_pruning(weight, block_size, sparse_ratio)
    print(result)

    result = vanilla_pruning(weight, sparse_ratio)
    print(result)

