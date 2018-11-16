import tensorflow as tf
import os
import random
import math
import sys
import time

#验证集数量
_NUM_TEST=300
#随机种子
_RANDOM_SED=0
#数据块
_NUM_SHARDS=2
#数据集路径
DATASETS_DIR=r'C:\Users\zhuha\PycharmProjects\digital_recognize\models-master\research\slim\images/'
#标签文件名字
LABEL_FILENAME=r'C:\Users\zhuha\PycharmProjects\digital_recognize\models-master\research\slim\images/labels.txt'

#定义tfrecords文件路径+名字
def _get_dataset_filename(dataset_dir,split_name,shard_id):
    output_filename='image_%s_%05d-of-%05d.tfrecord' % (split_name,shard_id,_NUM_SHARDS)
    return os.path.join(dataset_dir,output_filename)

#判断tfrecords是否存在
def _dataset_exists(dataset_dir):
    for split_name in['train','test']:
        for shard_id in range(_NUM_SHARDS):
            #定义tfrecord文件路径+名字
            output_filename=_get_dataset_filename(dataset_dir,split_name,shard_id)
        if not tf.gfile.Exists(output_filename):
            return False
    return True

#获取所有文件及分类
def _get_filenames_and_classes(dataset_dir):
    #数据目录
    directories=[]
    #分类名称
    class_names=[]
    for filename in os.listdir(dataset_dir):
        #合并文件路径
        path=os.path.join(dataset_dir,filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
            
    photo_filenames=[]
    for directory in directories:
        for filename in os.listdir(directory):
            path=os.path.join(directory,filename)
            photo_filenames.append(path)
    return photo_filenames,class_names

#字符串类型转换为bytesFeature
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

#int类型转换为intFeature
def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values=[values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def image_to_tfexample(image_data,image_format,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':bytes_feature(image_data),
        'image/format':bytes_feature(image_format),
        'image/class/label':int64_feature(class_id),
    }))

#把数据转换为tfrecord格式
def _convert_dataset(split_name,filenames,class_names_to_ids,dataset_dir):
    assert split_name in ['train','test']
    #计算每个数据块有多少数据
    num_per_shard=int(len(filenames)/_NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                #定义tfrecords文件路径+名字
                output_filename=_get_dataset_filename(dataset_dir,split_name,shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    #每一个数据块开始位置
                    start_ndx=shard_id*num_per_shard
                    #每一个数据块最后位置
                    end_ndx=min(num_per_shard*(shard_id+1),len(filenames))
                    
                    for i in range(start_ndx,end_ndx):
                        #使用try防止损坏的图片出错
                        try:
                            sys.stdout.write('\r>>Converting image %d/%d shard %d' %(i+1,len(filenames),shard_id))
                            sys.stdout.flush()
                            #读取图片
                            image_data=tf.gfile.GFile(filenames[i],'rb').read()
                            #获取图片类别
                            class_name=os.path.basename(os.path.dirname(filenames[i]))
                            class_id=class_names_to_ids[class_name]
                            #声称tfrecord文件
                            example=image_to_tfexample(image_data,b'jpg',class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print('could not read:',filenames[i])
                            print('Error:',e)
                            print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

#输出labels文件
def write_label_file(labels_to_class_names,dataset_dir,filename=LABEL_FILENAME):
    labels_filename=os.path.join(dataset_dir,filename)
    with tf.gfile.Open(labels_filename,'w') as f:
        for label in labels_to_class_names:
            class_name=labels_to_class_names[label]
            f.write('%d:%s\n' % (label,class_name))

if __name__=='__main__':
    #判断tfrecord文件是否存在
    if _dataset_exists(DATASETS_DIR):
        print('tfrecord文件已经存在')
    else:
        #获得所有图片及分类
        photo_finames,class_names=_get_filenames_and_classes(DATASETS_DIR)
        #分类转换为字典格式，类似于{'car':2,'guitar':1}
        class_names_to_ids=dict(zip(class_names,range(len(class_names))))
        
        #分割数据为训练集和测试集
        random.seed(_RANDOM_SED)
        random.shuffle(photo_finames)
        training_filenames=photo_finames[_NUM_TEST:]
        testing_filenames = photo_finames[:_NUM_TEST]
        
        #数据转换
        _convert_dataset('train',training_filenames,class_names_to_ids,DATASETS_DIR)
        _convert_dataset('test',testing_filenames,class_names_to_ids,DATASETS_DIR)
        
        #输出labels文件
        labels_to_class_names=dict(zip(range(len(class_names)),class_names))
        write_label_file(labels_to_class_names,DATASETS_DIR)
