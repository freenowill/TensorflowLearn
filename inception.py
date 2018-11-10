import tensorflow as tf
import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt

class NodeLookup(object):
    def __init__(self):
        label_lookup_path=r'C:\Users\zhuha\PycharmProjects\digital_recognize\inception_model\imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path=r'C:\Users\zhuha\PycharmProjects\digital_recognize\inception_model\imagenet_synset_to_human_label_map.txt'
        self.node_lookup=self.load(label_lookup_path,uid_lookup_path)
    
    def load(self,label_lookup_path,uid_lookup_path):
        #加载分类字符串n**********对应分类名称的文件
        uid_lines=tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human={}
        for line in uid_lines:
            #去掉换行符
            line=line.strip('\n')
            #按'\t'进行分割
            parsed_items=line.split('\t')
            #获取分类编号
            uid=parsed_items[0]
            #获取分类名称
            human_string=parsed_items[1]
            #保存编号字符串与分类名称的关系
            uid_to_human[uid]=human_string
            
        #加载分类字符串与分类编号的文件
        label_lines=tf.gfile.GFile(label_lookup_path).readlines()
        label_to_uid={}
        for line in label_lines:
            if line.startswith('  target_class:'):
                target_class=int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string=line.split(': ')[1]
                label_to_uid[target_class]=target_class_string[1:-2]
        print(label_to_uid)

        #建立编号与分类名的映射
        label_to_name={}
        for key,val in label_to_uid.items():
            #获取分类名称
            name=uid_to_human[val]
            label_to_name[key]=name
        return label_to_name
    
    #传入分类编号，返回分类名称
    def id_to_string(self,id):
        if id not in self.node_lookup:
            return ''
        return self.node_lookup[id]

#创建一个图存放google训练好的模型
with tf.gfile.FastGFile(r'C:\Users\zhuha\PycharmProjects\digital_recognize\inception_model\classify_image_graph_def.pb','rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')

with tf.Session() as sess:
    softmax_tensor=sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk(r'C:\Users\zhuha\PycharmProjects\digital_recognize\inception_model\image/'):
        for file in files:
            #载入图片
            image_data=tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            predictions=sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
            predictions=np.squeeze(predictions) #转为一维数据
            
            # #打印图片或名称
            # image_path=os.path.join(root,file)
            # print(image_path)
            # #显示图片
            # img=Image.open(image_path)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            
            #排序
            top_k=predictions.argsort()[-5:][::-1]
            node_lookup=NodeLookup()
            for id in top_k:
                #获取分类名称
                human_string=node_lookup.id_to_string(id)
                #获取置信度
                score=predictions[id]
                print('%s (score=%0.5f)'%(human_string,score))
            print()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        