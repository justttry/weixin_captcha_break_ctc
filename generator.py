#coding:utf-8
import random
import os
from itertools import product
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import unittest

"""
基本：
1 图片size
2 字符个数
3 字符区域（重叠、等分）
4 字符位置（固定、随机）
5 字符size（所占区域大小的百分比）
6 字符fonts
7 字符 type （数字、字母、汉字、数学符号）
8 字符颜色
9 背景颜色

高级：
10 字符旋转
11 字符扭曲
12 噪音（点、线段、圈）
"""
import string
chars = string.ascii_lowercase + string.ascii_uppercase

background = {'blue': (246, 246, 246),
              'green': (237, 247, 254),
              'orange': (237, 247, 254)}

def getTextColor(color):
    if color == 'green':
        red = random.randint(0, 50)
        blue = red + 60 + random.randint(-30, 30)
        green = blue + 50 + random.randint(-30, 30)
    elif color == 'blue':
        red = random.randint(0, 50)
        green = red + 50 + random.randint(-30, 30)
        blue = green + 170 + random.randint(-50, 50)
    else:
        blue = random.randint(0, 50)
        green = blue+70+random.randint(-30, 30)
        red = blue+125+random.randint(-30, 30)
    return blue, green, red

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([chars[x] for x in y])

#----------------------------------------------------------------------
def sin(x, height):
    """"""
    a = float(random.randint(12, 128))
    d = random.choice([2, 4])
    c = random.randint(1, 100)
    e = random.uniform(-0.1, 0.1)
    f = random.randint(0, 10) if e < 0 else random.randint(-10, 0)
    b = 2 * random.random()
    return np.array(map(int, height / d * (np.sin((x+c)/a) + b) * (e * x + f)))

def randRGB():
    return (random.randint(40, 110), random.randint(40, 110), random.randint(40, 110))

#----------------------------------------------------------------------
def movedown(img, c):
    """"""
    i, j, _ = img.shape
    a = img[:, :c]
    b = img[:, c:]
    d = np.array([[[0, 0, 0, 0]] * (j - c)])
    e = np.concatenate([d, b])[:-1]
    return np.concatenate([a, e], axis=1)

#----------------------------------------------------------------------
def moveup(img, c):
    """"""
    i, j, _ = img.shape
    a = img[:, :c]
    b = img[:, c:]
    d = np.array([[[0, 0, 0, 0]] * (j - c)])
    e = np.concatenate([b, d])[1:]
    return np.concatenate([a, e], axis=1)

#----------------------------------------------------------------------
def moveleft(img, c):
    """"""
    i, j, _ = img.shape
    a = img[:c]
    b = img[c:]
    d = np.array([[[0, 0, 0, 0]]] * (i - c))
    e = np.concatenate([b, d], axis=1)[:, 1:]
    return np.concatenate([a, e])

#----------------------------------------------------------------------
def moveright(img, c):
    """"""
    i, j, _ = img.shape
    a = img[:c]
    b = img[c:]
    d = np.array([[[0, 0, 0, 0]]] * (i - c))
    e = np.concatenate([d, b], axis=1)[:, :-1]
    return np.concatenate([a, e])

#----------------------------------------------------------------------
def move(im):
    """"""
    img = np.asarray(im)
    img = movemat(img)
    return Image.fromarray(img.astype('uint8'))

def movemat(img):
    rows, cols, _ = img.shape
    for i in range(1, rows):
        direction = random.choice([-1, 0, 0, 1])
        if direction == 1:
            img = np.concatenate([img, np.array([[[0, 0, 0, 0]]]*rows)], axis=1)
            img = moveright(img, i)
            if not any([any(j) for j in img[:, -1]]):
                img = img[:, :-1]
        elif direction == -1:
            img = np.concatenate([np.array([[[0, 0, 0, 0]]]*rows), img], axis=1)
            img = moveleft(img, i)
            if not any([any(j) for j in img[:, 0]]):
                img = img[:, 1:]
    rows, cols, _ = img.shape    
    for i in range(1, cols):
        direction = random.choice([-1, 0, 0, 1])
        if direction == 1:
            img = np.concatenate([np.array([[[0, 0, 0, 0]]*cols]), img])
            img = moveup(img, i)
            if not any([any(j) for j in img[0]]):
                img = img[1:]
        elif direction == -1:
            img = np.concatenate([img, np.array([[[0, 0, 0, 0]]*cols])])
            img = movedown(img, i)
            if not any([any(j) for j in img[-1]]):
                img = img[:-1]
    return img

def cha_draw(cha, text_color, font, rotate, size_cha, max_angle=15):
    im = Image.new(mode='RGBA', size=(size_cha*2, size_cha*2))
    drawer = ImageDraw.Draw(im) 
    drawer.text(xy=(0, 0), text=cha, fill=text_color, font=font) #text 内容，fill 颜色， font 字体（包括大小）
    if rotate:
        #max_angle = 45 # to be tuned
        angle = random.randint(-max_angle, max_angle)
        im = im.rotate(angle, Image.BILINEAR, expand=1)
    im = im.crop(im.getbbox())
    im = move(im)
    return im

def captcha_draw(size_im, nb_cha, set_cha, colors, fonts=None, overlap=0.01, 
        rd_bg_color=False, rd_text_color=False, rd_text_pos=False, rd_text_size=False,
        rotate=False, noise=None, dir_path=''):
    """
        overlap: 字符之间区域可重叠百分比, 重叠效果和图片宽度字符宽度有关
        字体大小 目前长宽认为一致！！！
        所有字大小一致
        扭曲暂未实现
        noise 可选：point, line , circle
        fonts 中分中文和英文字体
        label全保存在label.txt 中，文件第i行对应"i.jpg"的图片标签，i从1开始
    """
    rate_cha = 1.6 # rate to be tuned
    rate_noise = 0.25 # rate of noise
    cnt_noise = random.randint(10, 20)
    width_im, height_im = size_im
    width_cha = int(width_im / max(nb_cha-overlap, 1)) # 字符区域宽度
    height_cha = height_im # 字符区域高度
    width_noise = width_im
    height_noise = height_im
    bg_color = 'white'
    text_color = 'black'
    derx = 28
    dery = 8

    if rd_text_size:
        rate_cha = random.uniform(rate_cha-0.05, rate_cha+0.05) # to be tuned
    size_cha = int(rate_cha*min(width_cha, height_cha)) # 字符大小
    size_noise = int(rate_noise*height_noise) # 字符大小
    
    #if rd_bg_color:
        #bg_color = randRGB()
    color = random.choice(colors)
    bg_color = background[color] # 背景色
    im = Image.new(mode='RGB', size=size_im, color=bg_color) # color 背景颜色，size 图片大小

    drawer = ImageDraw.Draw(im)
    contents = []  
    
    text_color = getTextColor(color)
    
    first_x_point = random.randint(0, 10)
        
    for i in range(nb_cha):
        overlap_i = random.choice([0, overlap])
        # font = ImageFont.truetype("arial.ttf", size_cha)
        cha = random.choice(set_cha)
        font = random.choice(fonts)
        font = ImageFont.truetype(font, size_cha-5)
        contents.append(cha) 
        im_cha = cha_draw(cha, text_color, font, rotate, size_cha-5)
        im_cha_x, im_cha_y = im_cha.size
        im.paste(im_cha, 
                 (first_x_point, dery++random.randint(-2, 2)), 
                 im_cha) # 字符左上角位置
        first_x_point += im_cha_x - int(overlap_i*width_cha) if im_cha_x < 25 else \
            derx - int(overlap_i*width_cha)
        
    if 'sin' in noise:
        sincnt = random.randint(1, 2)
        for _ in range(sincnt):
            x = np.arange(0, width_im)
            y = sin(x, height_im)
            sinfill = random.choice([text_color, bg_color])
            cnt = random.randint(1, 3)
            for k in range(4):
                for i, j in zip(x, y+k):
                    drawer.point(xy=(i, j), fill=sinfill)
            
    return np.asarray(im), contents

def captcha_generator(width, 
                      height, 
                      batch_size=64,
                      set_cha=chars,
                      font_dir='/home/ubuntu/sina_captcha_fonts'
                      ):
    size_im = (width, height)
    rd_text_poss = [True, True]
    rd_text_sizes = [True, True]
    rd_text_colors = [True, True] # false 代表字体颜色全一致，但都是黑色
    rd_bg_color = True 
    noises = [['line', 'point', 'sin']]
    rotates = [True, True]
    nb_chas = [4, 6]
    font_paths = []
    for dirpath, dirnames, filenames in os.walk(font_dir):
        for filename in filenames:
            filepath = dirpath + os.sep + filename
            font_paths.append(filepath)
    
    rd_color = ['orange', 'blue', 'green']
    n_len = 4
    n_class = len(set_cha)
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]    
    while True:
        for i in range(batch_size):
            overlap = random.uniform(0, 0.1)
            rd_text_pos = random.choice(rd_text_poss)
            rd_text_size = random.choice(rd_text_sizes)
            rd_text_color = random.choice(rd_text_colors)
            noise = random.choice(noises)
            rotate = random.choice(rotates)
            nb_cha = 4
            dir_name = 'all'
            dir_path = 'img_data/'+dir_name+'/'
            im, contents = captcha_draw(size_im=size_im, nb_cha=nb_cha, set_cha=set_cha, 
                                        overlap=overlap, rd_text_pos=rd_text_pos, rd_text_size=True, 
                                        rd_text_color=rd_text_color, rd_bg_color=rd_bg_color, noise=noise, 
                                        rotate=rotate, dir_path=dir_path, fonts=font_paths, colors=rd_color)
            contents = ''.join(contents)
            X[i] = im
            for j, ch in enumerate(contents):
                y[j][i, :] = 0
                y[j][i, set_cha.find(ch)] = 1
        yield X, y  

def ctc_captcha_generator(width,
                  height,
                  conv_shape,
                  batch_size=64,
                  set_cha=chars,
                  font_dir='/home/ubuntu/sina_captcha_fonts'
                  ):
    size_im = (width, height)
    overlaps = [0.0, 0.3, 0.6]
    rd_text_poss = [True, True]
    rd_text_sizes = [True, True]
    rd_text_colors = [True, True] # false 代表字体颜色全一致，但都是黑色
    rd_bg_color = True 
    noises = [['line', 'point', 'sin']]
    rotates = [True, True]
    nb_chas = [4, 6]
    font_paths = []
    for dirpath, dirnames, filenames in os.walk(font_dir):
        for filename in filenames:
            filepath = dirpath + os.sep + filename
            font_paths.append(filepath)
    
    rd_color = ['orange', 'blue', 'green']
    n_len = 4
    n_class = len(set_cha)
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        for i in range(batch_size):
            overlap = random.uniform(0, 0.1)
            rd_text_pos = random.choice(rd_text_poss)
            rd_text_size = random.choice(rd_text_sizes)
            rd_text_color = random.choice(rd_text_colors)
            noise = random.choice(noises)
            rotate = random.choice(rotates)
            nb_cha = 4
            dir_name = 'all'
            dir_path = 'img_data/'+dir_name+'/'
            im, contents = captcha_draw(size_im=size_im, nb_cha=nb_cha, set_cha=set_cha, 
                                        overlap=overlap, rd_text_pos=rd_text_pos, rd_text_size=True, 
                                        rd_text_color=rd_text_color, rd_bg_color=rd_bg_color, noise=noise, 
                                        rotate=rotate, dir_path=dir_path, fonts=font_paths, colors=rd_color)
            contents = ''.join(contents)
            X[i] = im.transpose(1, 0, 2)
            y[i] = [chars.find(x) for x in contents]
        yield [X, y, np.ones(batch_size)*conv_shape, np.ones(batch_size)*n_len], np.ones(batch_size)
        
#----------------------------------------------------------------------
def captcha_save():
    """"""
    a = captcha_generator(130, 53)
    dir_path = 'img_data/all/'
    X, y = a.next()
    for x in X:
        if os.path.exists(dir_path) == False: # 如果文件夹不存在，则创建对应的文件夹
            os.makedirs(dir_path)
            pic_id = 1
        else:
            pic_names = map(lambda x: x.split('.')[0], os.listdir(dir_path))
            #pic_names.remove('label')
            pic_id = max(map(int, pic_names))+1 # 找到所有图片的最大标号，方便命名
        
        b, g, r = cv2.split(x)
        x = cv2.merge([r, g, b])
        img_name = str(pic_id) + '.jpg'
        img_path = dir_path + img_name
        label_path = dir_path + 'label.txt'
        #with open(label_path, 'a') as f:
            #f.write(''.join(pic_id)+'\n') # 在label文件末尾添加新图片的text内容
        print img_path
        img = Image.fromarray(np.uint8(x))
        img.save(img_path)            
        
#----------------------------------------------------------------------
def ctc_captcha_save():
    """"""
    a = ctc_captcha_generator(130, 53, 17)
    dir_path = 'img_data/all/'
    [X, y, _, _], _ = a.next()
    for x in X:
        x = x.transpose(1, 0, 2)
        if os.path.exists(dir_path) == False: # 如果文件夹不存在，则创建对应的文件夹
            os.makedirs(dir_path)
            pic_id = 1
        else:
            pic_names = map(lambda x: x.split('.')[0], os.listdir(dir_path))
            #pic_names.remove('label')
            pic_id = max(map(int, pic_names))+1 # 找到所有图片的最大标号，方便命名
        
        b, g, r = cv2.split(x)
        x = cv2.merge([r, g, b])    
        img_name = str(pic_id) + '.jpg'
        img_path = dir_path + img_name
        label_path = dir_path + 'label.txt'
        #with open(label_path, 'a') as f:
            #f.write(''.join(pic_id)+'\n') # 在label文件末尾添加新图片的text内容
        print img_path
        img = Image.fromarray(np.uint8(x))
        img.save(img_path)            
        
########################################################################
class GeneratorTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def setUp(self):
        """Constructor"""
        self.a = np.zeros((4, 3, 3))
        for i in range(3):
            self.a[i, i, i] = i+1
    
    #----------------------------------------------------------------------
    def test_movedown(self):
        """"""
        a = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        b = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]]])
        c = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]]])
        d = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]]])
        self.assertListEqual(self.a.tolist(), a.tolist())
        self.assertListEqual(movedown(self.a, 0).tolist(), b.tolist())
        self.assertListEqual(movedown(self.a, 1).tolist(), c.tolist())
        self.assertListEqual(movedown(self.a, 2).tolist(), d.tolist())
    
    #----------------------------------------------------------------------
    def test_moveup(self):
        """"""
        a = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        b = np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        c = np.array([[[1, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        d = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        self.assertListEqual(self.a.tolist(), a.tolist())
        self.assertListEqual(moveup(self.a, 0).tolist(), b.tolist())
        self.assertListEqual(moveup(self.a, 1).tolist(), c.tolist())
        self.assertListEqual(moveup(self.a, 2).tolist(), d.tolist())
        
    #----------------------------------------------------------------------
    def test_moveleft(self):
        """"""
        a = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        b = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 3], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        c = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 3], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        d = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 3], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        e = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        self.assertListEqual(self.a.tolist(), a.tolist())
        self.assertListEqual(moveleft(self.a, 0).tolist(), b.tolist())
        self.assertListEqual(moveleft(self.a, 1).tolist(), c.tolist())
        self.assertListEqual(moveleft(self.a, 2).tolist(), d.tolist())
        self.assertListEqual(moveleft(self.a, 3).tolist(), e.tolist())
        
    #----------------------------------------------------------------------
    def test_moveright(self):
        """"""
        a = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        b = np.array([[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 2, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        c = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 2, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        d = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        e = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        self.assertListEqual(self.a.tolist(), a.tolist())
        self.assertListEqual(moveright(self.a, 0).tolist(), b.tolist())
        self.assertListEqual(moveright(self.a, 1).tolist(), c.tolist())
        self.assertListEqual(moveright(self.a, 2).tolist(), d.tolist())
        self.assertListEqual(moveright(self.a, 3).tolist(), e.tolist())
        
    
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(GeneratorTest('test_movedown'))
    suite.addTest(GeneratorTest('test_moveup'))
    suite.addTest(GeneratorTest('test_moveleft'))
    suite.addTest(GeneratorTest('test_moveright'))
    return suite

if __name__ == "__main__":
    #unittest.main(defaultTest='suite')
    ctc_captcha_save()
