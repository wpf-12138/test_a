import tensorflow as tf
import numpy as np

# 添加层
def add_layer(inputs,in_size,out_size,activation_function=None)
    # add one more layer and return the output of this year
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
# 1.训练数据
# make up some real data
x_data=np.linspace(-1,1,300)[:,np.newaxis]
print(x_data)