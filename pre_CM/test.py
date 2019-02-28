import tensorflow as tf
import numpy as np

# A_1 = [[1,2,3],[4,5,6]]
# A_2 = [[4,5,6],[7,8,9]]
# print(type(A_1))
# B_1 = np.array(A_1, dtype = np.int32)
# B_2 = np.array(A_2, dtype = np.int32)
# print(type(B_1))

# C = []

# C.append(B_1)
# C.append(B_2)
# print(type(C))

# C_new = np.array(C, dtype = np.int32)

# B = tf.convert_to_tensor(C_new,dtype = tf.int32)
# B = tf.transpose(B, [2,0,1])

# with tf.Session() as sess:
# 	print(type(B))
# 	print(B.get_shape())


logits = [[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]
targets = [2,0,2,2]

correct_prediction = tf.equal(tf.argmax(logits, 1),tf.cast(targets, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	print(sess.run(accuracy))