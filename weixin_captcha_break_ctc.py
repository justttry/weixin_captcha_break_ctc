# encoding:UTF-8

import time
print 'author: justry'
print time.ctime()

# 导入库
import os
import string
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras import callbacks
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from distkeras.trainers import *
from distkeras.predictors import *
from distkeras.transformers import *
from distkeras.evaluators import *
from distkeras.utils import *
import distkeras.utils
from distkeras.job_deployment import Job

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext

from keras import backend as K
import tensorflow as tf   

from keras.objectives import custom_loss, ctc_lambda_func

# 建立SparkContext

application_name = "weixin captcha break"
master = 'yarn'
deploymode = 'client'
num_executors = 5
num_cores = 1
num_workers = num_executors * num_cores
optimizer = 'adagrad'
loss = 'categorical_crossentropy'

addition = 0
master_port = 5000
send_port = 8000
master_port += addition
send_port += addition
print master_port
print send_port

chars = string.ascii_lowercase + string.ascii_uppercase
width, height, n_len, n_class = 130, 53, 4, len(chars)+1

conf = SparkConf()
conf.set("spark.app.name", application_name)
conf.set("spark.master", master)
conf.set("spark.submit.deployMode", deploymode)
conf.set("spark.executor.cores", `num_cores`)
conf.set("spark.executor.instances", `num_executors`)
conf.set("spark.sql.warehouse.dir", "hdfs://master:9000/user/hive/warehouse");

###############################################################################
#from pyspark.sql import SparkSession
#sc = SparkSession.builder.master(master).appName(application_name).enableHiveSupport().getOrCreate()
#sqlContext = SQLContext(sc)
################################################################################
sc = SparkContext(conf=conf)
################################################################################

# 定义CTC模型，构造训练器
rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(4):
    x = Conv2D(32*2**i, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])

model.summary()

model.save('/home/ubuntu/models/weixin_ctc_rawmodel.h5')

from distkeras.job_deployment import graph
graph.append(tf.get_default_graph())

import time
print '------------------------test0----------------------------'
print time.ctime()

trainer = AEASGD(keras_model=model, worker_optimizer=optimizer, loss=custom_loss, num_workers=num_workers, 
            batch_size=32, features_col="features_normalized", label_col="newlabel", num_epoch=1,
            communication_window=32, rho=5.0, learning_rate=0.1, master_port=master_port)

print '------------------------test1----------------------------'
print time.ctime()

# 建立调度任务
job = Job("3Q20LA3MXU3N8Y9NVJ7A1T5WNHL2IWQSNNJ5V9I5P7MRJ8LSC33EN2DT3EWYLCJA",
          "user1",
          "data_path",
          5,
          1,
          trainer,
          3000,
          12)

print '------------------------test2----------------------------'
print time.ctime()

# 启动任务
job.send_with_files('http://13.124.171.155:%d'%send_port, ['generator.py'])

print '------------------------test3----------------------------'
print time.ctime()

# 等待结束
job.wait_completion()

print '------------------------test4----------------------------'
print time.ctime()

# 保存模型
trained_model = job.get_trained_model()
trained_model.save('weixin_ctc_trained_model.h5')

# 关闭sc
sc.stop()
