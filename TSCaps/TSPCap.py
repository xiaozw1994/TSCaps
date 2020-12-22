import os 
import numpy as np
import tensorflow as tf
import  TSCaps.config as cfg
import  time
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn import metrics



# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)




####
###   transform value to vector by using a squash activition
eposilion = 1e-9

def batch_normal(value,is_training=False,name='batch_norm'):
    if is_training is True:
         return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #Testing to Use
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)
def Active(x,mode='relu'):
    if mode == 'relu' :
        return tf.nn.relu(x)
    elif mode == 'leaky_relu' :
        return tf.nn.leaky_relu(x,alpha=0.1)
    else:
        return tf.nn.tanh(x)
mode = 'leaky_relu'
def CNNs(value,channel,kernel,stride,padding,is_training):
    conv = tf.contrib.layers.conv2d(value,channel,kernel,stride,padding=padding,activation_fn=None)
    bn = batch_normal(conv,is_training)
    res = Active(bn,mode)
    return res
def Res_block(x,channel,kernel, is_train):
    x1 = tf.contrib.layers.conv2d(x,channel,kernel,stride=1,padding='SAME',activation_fn=None)
    x1 = batch_normal(x1,is_train)
    x1 = Active(x1,mode)
    x2 = tf.contrib.layers.conv2d(x1,channel,kernel,stride=[2,1],padding="SAME",activation_fn=None)
    x2 = batch_normal(x2,is_train)
    x3 = tf.contrib.layers.conv2d(x,channel,1,stride=[2,1],padding="SAME",activation_fn=None)
    x3 = batch_normal(x3,is_train)
    x4 = x2 + x3
    return Active(x4,mode)
def Keep_residual(x,channel,kernel,is_train):
    x1 = tf.contrib.layers.conv2d(x,channel,kernel,stride=1,padding="SAME",activation_fn=None)
    x1 = batch_normal(x1,is_train)
    x1 = Active(x1,mode)
    x2 = tf.contrib.layers.conv2d(x1,channel,kernel,stride=1,padding="SAME",activation_fn=None)
    x2 = batch_normal(x2,is_train)
    x3 = x + x2 
    return Active(x3,mode)

def MutilCNNs(X,channel,stride,padding='SAME',is_train=True):
    head1 = tf.contrib.layers.conv2d(X,channel,[9,1],stride=stride,padding=padding,activation_fn=None)
    head2 = tf.contrib.layers.conv2d(X,channel,[7,1],stride=stride,padding=padding,activation_fn=None)
    head3 = tf.contrib.layers.conv2d(X,channel,[5,1],stride=stride,padding=padding,activation_fn=None)
    head4 = tf.contrib.layers.conv2d(X,channel,[3,1],stride=stride,padding=padding,activation_fn=None)
    head = tf.concat([head1,head2,head3,head4],axis=-1)
    head = batch_normal(head,is_train)
    residual = tf.contrib.layers.conv2d(X,channel,1,stride=stride,padding=padding,activation_fn=None)
    residual = batch_normal(residual,is_train)
    result = head + residual
    return Active(result,mode)




def squash(value):
    vec_square_norm = tf.reduce_sum(tf.square(value),-2,keepdims=True )
    scales = vec_square_norm / (1+ vec_square_norm) / tf.sqrt(vec_square_norm + eposilion)
    return value * scales
###
###
def layers_vector(vector,num_outputs,vec_len,kernel,strides,is_train,shapes):
    values = CNNs(vector,num_outputs*vec_len , kernel,strides,"SAME",is_train)
    values = tf.reshape(values,shapes)
    cap1 = squash(values)
    return cap1
## Capsule layers


def routing(input, b_IJ, num_outputs=6, num_dims=8):
    ''' The routing algorithm.

    Args:
        input:  shape, num_caps_l meaning the number of capsule in the layer l.
        num_outputs: the number of output capsules.
        num_dims: the number of dimensions for output capsule.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    input_shape = get_shape(input)
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
    # assert input.get_shape()

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    # assert u_hat.get_shape() 

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    n = 3  #3
    # line 3,for r iterations do
    for r_iter in range(n):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == n-1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() 

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape()
            elif r_iter < n-1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from
                # then matmul 
                # batch_size dim, resulting
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)

'''
Calculate the Next Step
'''
def getNextStride(step):
    if step %2 == 0:
        return step//2
    else :
        return step//2 +1
def DiscountStep(step,index):
    for i in range(index):
        step = getNextStride(step)
    return step
##########################
##############################
class CapMotion(object):
    def __init__(self,batch_size,length,num_label,decay,lr=0.0001,is_train=True ):
        #config = cfg.Config90()
        self.batch_size = batch_size
        self.decay = decay
        self.training = is_train
        self.length = length
        self.to_steps = 0
        self.next_length = DiscountStep(length,2)
        self.num_label = num_label
        self.x = tf.placeholder(dtype=tf.float32,shape=(None,self.length,1,1))
        self.y = tf.placeholder(dtype=tf.float32,shape = (None,self.num_label))
        self.is_train = tf.placeholder(dtype=tf.bool)
        self.cfg = cfg.Config()
        self.biuld_net()
        self.Single_acc()
        self.loss()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.RMadm = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

    def biuld_net(self):
       # gragh = tf.Graph()
       # with gragh.as_default():
                   ###########
                   ### set top conv
                 top_con = CNNs(self.x,128,[9,1],2,"SAME",self.is_train)
                 self.primary_cap = layers_vector(top_con,32,4,[9,1],2,self.is_train,shapes=[-1,self.next_length*8,16,1])
                 # [-1,88*16,8,1]
                    #with tf.variable_scope("capsules_layers"):
                 fc_function = tf.reshape(self.primary_cap,shape=(-1, self.primary_cap.shape[1].value,1, self.primary_cap.shape[-2].value,1))
                        #with tf.variable_scope("routing"):
                 #[-1,88*16,1,8,1]
                 blu = tf.constant( np.zeros([self.batch_size, self.primary_cap.shape[1].value,self.num_label,1,1]),dtype=tf.float32 )
                 caps = routing(fc_function,blu,num_outputs=self.num_label ,num_dims=32)
                 #### [120,37,8,1]
                 top_conv_1 = CNNs(self.x,128,[7,1],2,"SAME",self.is_train)
                 self.primary_cap_1 = layers_vector(top_conv_1,32,4,[7,1],2,self.is_train,shapes=[-1,self.next_length*16,8,1])
                 fc_function_1 = tf.reshape(self.primary_cap_1,shape=(-1,self.primary_cap_1.shape[1].value,1,self.primary_cap_1.shape[-2].value,1))
                 blu_1 = tf.constant(np.zeros([self.batch_size,self.primary_cap_1.shape[1].value,self.num_label,1,1]),dtype=tf.float32)
                 with  tf.variable_scope("routint_1"):
                     caps_1 = routing(fc_function_1,blu_1,self.num_label,16)
                 top_con_2 = CNNs(self.x,128,[5,1],2,'SAME',self.is_train)
                 self.primary_cap_2 = layers_vector(top_con_2,32,4,[5,1],2,self.is_train,shapes=[-1,self.next_length*32,4,1])
                 fc_function_2 = tf.reshape(self.primary_cap_2,shape=(-1,self.primary_cap_2.shape[1].value,1,self.primary_cap_2.shape[-2].value,1))
                 blu_2 = tf.constant(np.zeros([self.batch_size,self.primary_cap_2.shape[1].value,self.num_label,1,1]),dtype=tf.float32)
                 with tf.variable_scope("routing_2"):
                     caps_2 = routing(fc_function_2,blu_2,self.num_label,8)
                 
                 a = 3.0
                 b = 1.0
                 c = 1.0
                 #  a = 3.0
                 #  b = 1.0
                 caps = tf.concat([a*caps,b*caps_1,c*caps_2],axis=3)
                 # This is the best performance in our experiments.
                 
                 self.caps = tf.squeeze(caps,axis=1)
                 v_length = tf.sqrt(reduce_sum(tf.square(self.caps),axis=2,keepdims=True)+eposilion)
                 softmax_v = softmax(v_length,axis=1)
                #########[batch_size,num_label,1,1]
                 argmax_idx = tf.to_int32(tf.argmax(softmax_v,axis=1))
                 self.argmax_idx = tf.reshape(argmax_idx,shape=(self.batch_size,))
                ###
                 self.masked_v = tf.multiply(tf.squeeze(self.caps),tf.reshape(self.y,(-1,self.num_label,1)))
                 self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps),axis=2,keepdims=True)+eposilion)
                ########
                # decoder
                 vector_j = tf.reshape(self.masked_v,shape=(self.batch_size,-1))
                 fc1 = tf.contrib.layers.fully_connected(vector_j,num_outputs=256)
                 fc1 = tf.contrib.layers.fully_connected(fc1,num_outputs=512)
                 self.decode = tf.contrib.layers.fully_connected(fc1,num_outputs=self.length,activation_fn=tf.sigmoid)

    def loss(self):
        ###########
        max_l = tf.square(tf.maximum(0.0,self.cfg.init_seq-self.v_length))
        max_r = tf.square(tf.maximum(0.0,self.v_length-self.cfg.init_sub))
        max_l = tf.reshape(max_l,shape=(self.batch_size,-1))
        max_r = tf.reshape(max_r,shape=(self.batch_size,-1))
        true_label = self.y
        self.local_loss = tf.reduce_mean( tf.reduce_sum(true_label * max_l  + self.cfg.lamdaset * (1-true_label) * max_r,axis=1))
        orign = tf.reshape(self.x,(self.batch_size,-1))
        squared = tf.square(self.decode-orign)
        self.margin_loss = tf.reduce_mean(squared)
        self.total_loss = self.local_loss + self.decay* self.margin_loss
        #self.total_loss = self.local_loss
    ############################
    ###    Compute the number of accuracy objects
    ###############
    def Single_acc(self):
        correct_prediction =  tf.equal(  tf.to_int32(tf.argmax(self.y,axis=1)),self.argmax_idx  )
        self.acc_num = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
def Totalcount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
        # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("The Total params:",total_parameters/1e6,"M")

