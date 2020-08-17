
from modules import *
tf.enable_eager_execution()
import numpy
# event_file = './T-pred-Dataset/lastfm-v5k_event.txt'
# time_file = './T-pred-Dataset/lastfm-v5k_time2.txt'
#
# train_data, valid_data, test_data = read_data.data_split(event_file, time_file, shuffle=True)
# input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(valid_data,20,5)
# a,b,c,d = read_data.generate_batch(100,input_event_data,target_event_data,input_time_data,target_time_data)

a = np.array([[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]])
b = np.array([[1,1],[2,2],[3,3]])
c = []
for a_i, b_i in zip(a,b):
    print(b_i)
    print(a_i)
    c_i = tf.multiply(b_i, a_i)
    c.append(c_i)
c = tf.reduce_sum(c, axis= 1)

a = np.array([1,1,1,2,2,2,3,3,3,4],dtype=float)
b = np.array([2,2,2,2,2,2,2,2,2,2],dtype=float)

m = tf.keras.metrics.Mean()
m.update_state([1, 3, 5, 7])
print('Final result: ', m.result().numpy())

with tf.Session() as sess:
    #print(sess.run(tf.math.abs(100*a-100*b)))
    #print(sess.run(tf.reduce_mean(tf.math.abs(100*a-100*b))))
    #print(sess.run(tf.losses.huber_loss(100*a,100*b,delta=10.0)))
    # print(sess.run(tf.sign(tf.cast(tf.sequence_mask(3), tf.float32))))
    # print(sess.run(tf.sign([1,1,0])))
    # print(sess.run(gumbel_softmax(tf.constant([20,1,-4], dtype=tf.float32),tau=1)))
    print(sess.run(tf.one_hot([[1,1,1],[2,2,2],[3,3,3]],depth=4)))