import tensorflow as tf
import numpy as np
import  time
import os 
import TSCaps.data as da
import TSCaps.config as cfg
import TSCaps.TSPCap as model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
###
###          Load datasets
###################
##
###     
##########
class_name = cfg.training_name
length = cfg.length
sota = 0.9
data_dir = os.path.join(cfg.data_files, class_name)
x_train , y_train = da.readucr(data_dir+"/"+class_name+"_TRAIN")
x_test, y_test = da.readucr(data_dir+"/"+class_name+"_TEST")
### Get NumClasses
num_classes = da.GetNumClasses(y_test)
######## Label Normalization
y_train, y_test = da.NormalizationClassification(y_train,num_classes), da.NormalizationClassification(y_test,num_classes)
Y_train = da.OneHot(y_train,num_classes)
Y_test = da.OneHot(y_test,num_classes)
############### 
batch_size = 30

x_train = da.NormalizationFeatures(x_train).reshape((-1,length,1,1))
x_test = da.NormalizationFeatures(x_test).reshape((-1,length,1,1))
decay = 0.0005 * length

caps = model.CapMotion(batch_size,length,num_classes,decay)
total_train = x_train.shape[0] // batch_size
epoch = 150
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#saver = tf.train.Saver(tf.global_variables())
model.Totalcount()
value = 0
start_time = time.time()

for i in range(epoch):
    for j in range(total_train):
        batch_x = x_train[batch_size*j:(j+1)*batch_size,...]
        batch_y = Y_train[batch_size*j:(j+1)*batch_size,...]
        _, total_loss = sess.run(
            [caps.train_op,caps.total_loss],feed_dict={caps.x:batch_x,caps.y:batch_y,caps.is_train:True}
        )
    if i %1 == 0:
        acc = sess.run(
            caps.acc_num,feed_dict={ caps.x:x_test,caps.y:Y_test,caps.is_train:False}
        ) / x_test.shape[0]
        print("Epoch:%d =======loss:%.4f========acc:%.4f"%(i+1,total_loss,acc))
sess.close()









