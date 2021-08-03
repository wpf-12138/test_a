import tensorflow as tf
import numpy as np

# 添加层
def add_layer(inputs,in_size,out_size,activation_function=None):
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
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) -0.5 + noise

# 2.定义节点准备接受数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 3.定义神经层，隐藏层和预测层
# add hidden layer 输入值是xs，在隐藏层有10个神经元
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer 输入值是隐藏层l1,在预测层输出结果1
prediction = add_layer(l1,10,1,activation_function=None)
# 4.定义loss表达式
#the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices[1]))
# 5.选择optimizer使loss
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# import step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess=tf.Session()
# 上面的定义都没有运算，直到sess.run才会运算
sess.run(init)

# 迭代1000次学习,sess.run optimizer
for i in range(1000):
    #training train_step 和loss都是由placeholder定义的运算，所以这里要用feed传入参数
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 ==0:
        #to see the step improvement