import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#分批次数据
def get_batch(data,label,batch_size):
    input_deque=tf.train.slice_input_producer([data,label],num_epochs=10,shuffle=False,capacity=32)
    train_x_batch,label_x_batch=tf.train.batch(input_deque,batch_size=batch_size,num_threads=2,capacity=32,allow_smaller_final_batch=False)
    return train_x_batch,label_x_batch
#参数摘要
def varable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        stddev=tf.square(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

#载入数据集
train=pd.read_csv(r'C:\Users\zhuha\.kaggle\competitions\kaggle competitions download -c digit-recognizer\train.csv')[:10000]
label=train['label']
train.drop(['label'],axis=1,inplace=True)
train/=255.0
train_x,train_y,label_x,label_y=train_test_split(train,label,stratify=label)

label_x=label_x.reshape(-1,1)
label_y=label_y.reshape(-1,1)

#对label进行onehot编码
label_x=OneHotEncoder().fit(label_x).transform(label_x).toarray()
label_y=OneHotEncoder().fit(label_y).transform(label_y).toarray()

#分批次训练数据
train_x_batch,label_x_batch=get_batch(train_x,label_x,batch_size=100)
train_y_batch,label_y_batch=get_batch(train_y,label_y,batch_size=100)
#初始化权值及偏置值
def weight_variables(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_varible(shape):
    initial=tf.constant(0.1,shape=shape)
    return initial

#卷积层
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #ksize窗口大小

#定义placeholder
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x_input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')
    x_image=tf.reshape(x,[-1,28,28,1]) #1代表一维通道，彩色则为3

#初始化第一个卷积层的权值与偏置
with tf.name_scope('conv1'):
    with tf.name_scope('w_conv1'):
        w_conv1=weight_variables([5,5,1,32]) #1代表通道数1维，彩色则为3,32代表卷积核数目,5X5进行采样
        varable_summaries(w_conv1)
    with tf.name_scope('b_conv1'):
        b_conv1=bias_varible([32]) #1个卷积核则1个偏置值
        varable_summaries(b_conv1)

#对x_image进行第一层卷积再池化
    with tf.name_scope('relu1'):
        h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
    with tf.name_scope('pool1'):
        h_pool1=max_pool_2x2(h_conv1) #14x14

#初始化第二个卷积层的权值与偏置
with tf.name_scope('conv2'):
    with tf.name_scope('w_conv2'):
        w_conv2=weight_variables([5,5,32,64])
        varable_summaries(w_conv2)
    with tf.name_scope('b_conv2'):
        b_conv2=bias_varible([64])
        varable_summaries(b_conv2)
        
#第二层卷积与池化
    with tf.name_scope('relu2'):
        h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    with tf.name_scope('pool2'):
        h_pool2=max_pool_2x2(h_conv2) #7x7

#最后得到64张7x7的平面

#初始化全连接层
with tf.name_scope('fc1'):
    with tf.name_scope('w_fc1'):
        w_fc1=weight_variables([7*7*64,1024]) #上一层7*7*64个神经元，全连接层有1024个神经元
        varable_summaries(w_fc1)
    with tf.name_scope('b_fc1'):
        b_fc1=bias_varible([1024])
        varable_summaries(b_fc1)

#把池化层2输出扁平化为1维
    with tf.name_scope('pool2_flat'):
        h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64],name='h_pool2_flat')
#求第一个全连接层输出
    with tf.name_scope('fc1_out'):
        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
    with tf.name_scope('fc1_dropout'):
#使用dropout
        keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob,name='h_fc1_drop')

#初始化第二个全连接层
with tf.name_scope('fc2'):
    with tf.name_scope('w_fc2'):
        w_fc2=weight_variables([1024,10])
        varable_summaries(w_fc2)
    with tf.name_scope('b_fc2'):
        b_fc2=bias_varible([10])
        varable_summaries(b_fc2)

#计算输出
    with tf.name_scope('prediction'):
        prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#定义交叉熵代价函数
with tf.name_scope('loss'):
    cross_entrop=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',cross_entrop)
#使用AdamOptimizer优化
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entrop)
#预测结果
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#求准确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    
merged=tf.summary.merge_all()

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    train_writer = tf.summary.FileWriter('C:/Users/zhuha/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('C:/Users/zhuha/logs/test', sess.graph)
    
    # 开启协调器
    coord = tf.train.Coordinator()
    # 启用start_queue_runners
    threads = tf.train.start_queue_runners(sess, coord)
    epoch = 0
    try:
        while not coord.should_stop():
            batch_x, batch_x_label = sess.run([train_x_batch, label_x_batch])
            sess.run(train_step,feed_dict={x:batch_x,y:batch_x_label,keep_prob:0.7})
            #记录训练集计算的参数
            summary=sess.run(merged, feed_dict={x: batch_x, y: batch_x_label, keep_prob:1})
            train_writer.add_summary(summary,epoch)

            batch_y, batch_y_label = sess.run([train_y_batch, label_y_batch])
            #记录测试集计算的参数
            summary=sess.run(merged,feed_dict={x:batch_y,y:batch_y_label,keep_prob:1})
            acc=sess.run(accuracy,feed_dict={x:batch_y,y:batch_y_label,keep_prob:1})
            test_writer.add_summary(summary,epoch)
            
            epoch+=1
            print('Iter'+str(epoch)+','+'accuracy'+str(acc))
    except tf.errors.OutOfRangeError:
        print('Train end!')
    finally:
        coord.request_stop()
        print('program end!')
    coord.join(threads)