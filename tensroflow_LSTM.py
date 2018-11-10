import tensorflow as tf
import pandas as pd
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
train/=255.0
train_x,train_y,label_x,label_y=train_test_split(train,label,test_size=0.4)

label_x=label_x.reshape(-1,1)
label_y=label_y.reshape(-1,1)

#对label进行onehot编码
label_x=OneHotEncoder().fit(label_x).transform(label_x).toarray()
label_y=OneHotEncoder().fit(label_y).transform(label_y).toarray()

#分批次训练数据
train_x_batch,label_x_batch=get_batch(train_x,label_x,batch_size=100)
train_y_batch,label_y_batch=get_batch(train_y,label_y,batch_size=100)

#输入图片为28x28
n_inputs=28 #输入一行
max_time=28 #共28行
lstm_size=100 #隐层单元
n_class=10 #分类数目
batch_size=100 #每次训练100
n_batch=train_x_batch.shape[0]//100

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#初始化权值与偏置值
weights=tf.Variable(tf.truncated_normal([lstm_size,n_class],stddev=0.1))
biases=tf.Variable(tf.constant(0.1,shape=[n_class]))

#定义RNN
def RNN(X):
    inputs=tf.reshape(X,[-1,max_time,n_inputs])
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results

#计算RNN返回结果
prediction=RNN(x)
#损失函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#优化器
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
#求准确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    #开启协调器
    coord=tf.train.Coordinator()
    #开启start_queue_runner
    threads=tf.train.start_queue_runners(sess,coord)
    
    epoch=0
    try:
        while not coord.should_stop():
            batch_x_train,batch_x_label=sess.run([train_x_batch, label_x_batch])
            batch_y_train,batch_y_label=sess.run([train_y_batch,label_y_batch])
            
            sess.run(train_step,feed_dict={x:batch_x_train,y:batch_x_label})
            
            acc=sess.run(accuracy,feed_dict={x:batch_y_train,y:batch_y_label})
            epoch+=1
            if epoch%500==0:
                print('Iter'+str(epoch)+'accuracy'+str(acc))
    except tf.errors.OutOfRangeError:
        print('Train end!')
    finally:
        coord.request_stop()
        print('program end!')
    coord.join(threads)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

