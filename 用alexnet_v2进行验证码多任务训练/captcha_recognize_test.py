import tensorflow as tf
import os
from PIL import Image
import numpy as np
from nets import nets_factory

#字符数量
CHAR_SET_LEN=10
#图片高度
IMAGE_HEIGHT=60
#图片宽度
IMAGE_WIDTH=160
#批次
BATCH_SIZE=1
#TFRecord路径
TFRECORD_FILE='C:/Users/zhuha/PycharmProjects/digital_recognize/captcha/test.tfrecords'

#placeholder
x=tf.placeholder(tf.float32,[None,224,224])

##从tfrecord读出数据
def read_and_decode(filename):
    filename_queue=tf.train.string_input_producer([filename])
    reader=tf.TFRecordReader()
    
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,
                                     features={'image':tf.FixedLenFeature([],tf.string),
                                               'label0':tf.FixedLenFeature([],tf.int64),
                                               'label1':tf.FixedLenFeature([],tf.int64),
                                               'label2':tf.FixedLenFeature([],tf.int64),
                                               'label3':tf.FixedLenFeature([],tf.int64),
                                     })
    #获取图片数据
    image=tf.decode_raw(features['image'],tf.uint8)
    #没经过处理的原始图
    image_raw=tf.reshape(image,[224,224])
    #shuffle需要reshape
    image=tf.reshape(image,[224,224])
    #图片预处理
    image=tf.cast(image,tf.float32)/255.0
    image=tf.subtract(image,0.5)
    image=tf.multiply(image,2)
    #获取label
    label0=tf.cast(features['label0'],tf.int32)
    label1=tf.cast(features['label1'],tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    
    return image,image_raw,label0,label1,label2,label3

#读取图片
image,image_raw,label0,label1,label2,label3=read_and_decode(TFRECORD_FILE)

image_batch,image_raw_batch,label0_batch,\
label1_batch,label2_batch,label3_batch=tf.train.shuffle_batch([image,
                                                                image_raw,
                                                                label0,
                                                                label1,
                                                                label2,
                                                                label3],
                                                                batch_size=BATCH_SIZE,
                                                                capacity=5000,
                                                                min_after_dequeue=1000,
                                                                num_threads=1)
#定义网络结构
train_network_fn=nets_factory.get_network_fn('alexnet_v2',
                                             num_classes=CHAR_SET_LEN,
                                             weight_decay=0.0005,
                                             is_training=False)

config = tf.ConfigProto(allow_soft_placement=True)
#最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)
    
    #预测值
    prediction0=tf.reshape(logits0,[-1,CHAR_SET_LEN])
    prediction0=tf.argmax(prediction0,1)
    
    prediction1=tf.reshape(logits1,[-1,CHAR_SET_LEN])
    prediction1=tf.argmax(prediction1,1)
    
    prediction2=tf.reshape(logits2,[-1,CHAR_SET_LEN])
    prediction2=tf.argmax(prediction2,1)
    
    prediction3=tf.reshape(logits3,[-1,CHAR_SET_LEN])
    prediction3=tf.argmax(prediction3,1)
    
    #初始化
    sess.run(tf.global_variables_initializer())
    #载入模型
    saver=tf.train.Saver()
    saver.restore(sess,'C:/Users/zhuha/PycharmProjects/digital_recognize/captcha/model/captcha_recognize.model-6000')
    
    #创建协调器
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
    for i in range(10):
        b_image,b_image_raw,b_label0,b_label1,b_label2,b_label3=sess.run([image_batch,
                                                                          image_raw_batch,
                                                                          label0_batch,
                                                                          label1_batch,
                                                                          label2_batch,
                                                                          label3_batch])
        print('label:',b_label0,b_label1,b_label2,b_label3)
        
        #预测
        label0,label1,label2,label3=sess.run([prediction0,prediction1,prediction2,prediction3],
                                              feed_dict={x:b_image})
        #打印预测
        print('prediction',label0,label1,label2,label3)
    
    #关闭线程
    coord.request_stop()
    #其他线程关闭后，函数返回
    coord.join(threads)