import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#分批次数据
def get_batch(data,label,batch_size):
    input_deque=tf.train.slice_input_producer([data,label],num_epochs=10,shuffle=False,capacity=32)
    train_x_batch,label_x_batch=tf.train.batch(input_deque,batch_size=batch_size,num_threads=1,capacity=32,allow_smaller_final_batch=True)
    return train_x_batch,label_x_batch

#载入数据集
train=pd.read_csv(r'C:\Users\zhuha\.kaggle\competitions\kaggle competitions download -c digit-recognizer\train.csv')
label=train['label']
train.drop(['label'],axis=1,inplace=True)
train_x,train_y,label_x,label_y=train_test_split(train,label,stratify=label)

label_x=label_x.reshape(-1,1)
label_y=label_y.reshape(-1,1)

#对label进行onehot编码
label_x=OneHotEncoder().fit(label_x).transform(label_x).toarray()
label_y=OneHotEncoder().fit(label_y).transform(label_y).toarray()

#参数摘要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean) #平均值
        stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

#定义命名空间
with tf.name_scope('input'):
    #定义placeholder
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')
    keep_prob=tf.placeholder(tf.float32)
    lr=tf.Variable(0.001,dtype=tf.float32)

#定义神经网络输入层
with tf.name_scope('L1_layer'):
    with tf.name_scope('W_L1'):
        W_L1=tf.Variable(tf.truncated_normal([784,100],stddev=0.1))
        variable_summaries(W_L1)
    with tf.name_scope('b_L1'):
        b_L1=tf.Variable(tf.zeros([100])+0.1)
        variable_summaries(b_L1)
    with tf.name_scope('wx_plus_b_L1'):
        wx_plus_b_L1=tf.matmul(x,W_L1)+b_L1
    with tf.name_scope('tanh'):
        L1=tf.nn.tanh(wx_plus_b_L1)
    with tf.name_scope('dropout'):
        L1_dropout=tf.nn.dropout(L1,keep_prob)

#定义神经网络隐藏层
with tf.name_scope('L2_layer'):
    with tf.name_scope('W_L2'):
        W_L2=tf.Variable(tf.truncated_normal([100,50],stddev=0.1))
        variable_summaries(W_L2)
    with tf.name_scope('b_L2'):
        b_L2=tf.Variable(tf.zeros([50])+0.1)
        variable_summaries(b_L2)
    with tf.name_scope('wx_plus_b_L2'):
        wx_plus_b_L2=tf.matmul(L1,W_L2)+b_L2
    with tf.name_scope('tanh'):
        L2=tf.nn.tanh(wx_plus_b_L2)
    with tf.name_scope('dropout'):
        L2_dropout=tf.nn.dropout(L2,keep_prob)

#定义神经网络隐藏层
with tf.name_scope('L3_layer'):
    with tf.name_scope('W_L3'):
        W_L3=tf.Variable(tf.truncated_normal([50,20],stddev=0.1))
        variable_summaries(W_L3)
    with tf.name_scope('b_L3'):
        b_L3=tf.Variable(tf.zeros([20])+0.1)
        variable_summaries(b_L3)
    with tf.name_scope('wx_plus_b_L3'):
        wx_plus_b_L3=tf.matmul(L2,W_L3)+b_L3
    with tf.name_scope('tanh'):
        L3=tf.nn.tanh(wx_plus_b_L3)
    with tf.name_scope('dropout'):
        L3_dropout=tf.nn.dropout(L3,keep_prob)
#定义神经网络输出层
with tf.name_scope('L4_layer'):
    with tf.name_scope('W_L4'):
        W_L4=tf.Variable(tf.truncated_normal([20,10],stddev=0.1))
        variable_summaries(W_L4)
    with tf.name_scope('b_L4'):
        b_L4=tf.Variable(tf.zeros([10])+0.1)
        variable_summaries(b_L4)
    with tf.name_scope('wx_plus_b_L4'):
        wx_plus_b_L4=tf.matmul(L3,W_L4)+b_L4
    with tf.name_scope('prediction'):
        prediction=tf.nn.softmax(wx_plus_b_L4)

#定义代价函数
with tf.name_scope('loss'):
# loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_x,logits=prediction))
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
    
#使用梯度下降法
with tf.name_scope('train'):
# train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    train_step=tf.train.AdamOptimizer(lr).minimize(loss)

#求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)
with tf.name_scope('batch'):
    train_x_batch,label_x_batch=get_batch(train_x,label_x,batch_size=100)

merged=tf.summary.merge_all()

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    writer=tf.summary.FileWriter(r'C:\Users\zhuha\logs/',sess.graph)
    #开启协调器
    coord=tf.train.Coordinator()
    #启用start_queue_runners
    threads=tf.train.start_queue_runners(sess,coord)
    epoch=0
    try:
        while not coord.should_stop():
            sess.run(tf.assign(lr, 0.1 * (0.95 ** epoch//1000)))
            data,label=sess.run([train_x_batch,label_x_batch])
            summary,_=sess.run([merged,train_step],feed_dict={x:data,y:label,keep_prob:0.7})
            writer.add_summary(summary,epoch)
            acc=sess.run(accuracy,feed_dict={x:train_y,y:label_y,keep_prob:0.7})
            acc_train=sess.run(accuracy,feed_dict={x:train_x,y:label_x,keep_prob:0.7})
            print('Iter:'+str(epoch)+'test accuracy'+str(acc)+','+'train accuracy'+str(acc_train))
            epoch+=1
    except tf.errors.OutOfRangeError:
        print('Train end!')
    finally:
        coord.request_stop()
        print('program end!')
    coord.join(threads)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    