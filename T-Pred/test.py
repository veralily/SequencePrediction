import os
# import tensorflow as tf
import numpy as np
# from tqdm import tqdm
import time

# with tf.name_scope('cltdevelop'):
#     def uniform(stdev, size):
#     	return np.random.uniform(low = -stdev * np.sqrt(3),high = stdev * np.sqrt(3),size = size).astype('float32')
#     filter_values = uniform(1.0,(5, 5, 5))
#     print(filter_values.name)

# with tf.variable_scope('cltdevelop1'):
#     var_1 = tf.get_variable('var_1', shape=[1, ])
#     print(var_1.name)
# with tf.variable_scope('cltdevelop1'):    
#     var_1 = tf.get_variable('var_2', shape=[1, ])
#     with tf.variable_scope('cltdevelop1.2', reuse = tf.AUTO_REUSE):    
#     	var_2 = tf.get_variable('var_1', shape=[1, ])
#     	print(var_2)
#     print(var_1.name)
# with tf.variable_scope('cltdevelop2', reuse = tf.AUTO_REUSE):    
#     var_2 = tf.get_variable('var_1', shape=[1, ])
#     print(var_2.name)

# ## just name of tensors
# # def param_name():
# # 	variable_name = [v.name for v in tf.trainable_variables()]
# # 	return variable_name

# # print(param_name())

# for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#     print i

x = [[1,2,3],[4,5,6],[7,8,9]]
y = np.array(x)

a_list = list(range(20))
b_list = list(a_list)
np.random.shuffle(b_list)

def iterator(a_list):
	for i in a_list:
		yield (i+1)

g_1 = iterator(a_list)
# print(list(g_1))
g_2 = iterator(b_list)
# print(list(g_2))

s = 20 // 3
print('----------s:%d---------' % s)

for i in range(s):
    #print(next(gen))
    for i in range(3):
    	print('g_1:----%d' % g_1.next())
    print('g_2:----%d' % g_2.next())

# event_file = './T-pred-Dataset/IPTV_event.txt'
# time_file = './T-pred-Dataset/IPTV_time.txt'
# with open(event_file) as f:
# 	e_lines = f.readlines()
# 	with open(time_file) as f_1:
# 		t_lines = f_1.readlines()
# 		print(len(e_lines))
# 		print(len(t_lines))
# 		for i,(e,t) in enumerate(zip(e_lines, t_lines)):
# 			print(str(i) + ' e: ' + str(len(e.split('\t'))) + ' t: ' + str(len(t.split('\t'))))

# x = [[[0,0],[0,0],[0,0]],[[1,1],[1,1],[1,1]],[[2,2],[2,2],[2,2]]]
# y = [[5,1,1],[5,1,1],[5,1,1]]

# z = tf.concat([x, tf.expand_dims(y, 2)],2)

# print(z[:,0,:].get_shape())

# with tf.Session() as sess:
#     print(sess.run(z[:,0,:]))