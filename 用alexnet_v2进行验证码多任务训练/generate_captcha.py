from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import sys
import random

number=['0','1','2','3','4','5','6','7','8','9']

def random_captcha_text(char_set=number,captcha_size=4):
    #验证码列表
    captcha_text=[]
    for i in range(captcha_size):
        #随机选择
        c=random.choice(char_set)
        #加入验证码集
        captcha_text.append(c)
    return captcha_text
#生成验证码
def gen_captcha_text_and_image():
    image=ImageCaptcha()
    #获得随机生成验证码
    captcha_text=random_captcha_text()
    #转换成字符串
    captcha_text=''.join(captcha_text)
    #生成验证码
    image.generate(captcha_text)
    #写入文件
    image.write(captcha_text,'C:/Users/zhuha/PycharmProjects/digital_recognize/captcha/images/'+captcha_text+'.jpg')

#实际数量少于10000
num=10000
if __name__=='__main__':
    for i in range(num):
        gen_captcha_text_and_image()
        print('\r>>Creating image %d/%d' % (i+1,num))
    print('\n')
    print('生成完毕')