import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定义样本
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义中间层10个神经元
Weight_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.random_normal([1,10]))
Wx_plus_b_L1=tf.matmul(x,Weight_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1)

#定义输出层
Weight_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.random_normal([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weight_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)

#定义代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for _ in range(2000):
        sess.run(train_step,feed_dict={y:y_data,x:x_data})
    #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    #画图
    fig=plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()


