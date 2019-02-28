# def generator(length = 10):
# 	for i in range(length):
# 		yield i

# gen = generator(20)

# for i in range(10):
# 	print(gen.next())


import reader_event_sequence as reader
import tensorflow as tf

'''
a = [[1,2,3],[4,5,6],[7,8,9]]
c = zip(*a)
print(c)
'''

'''
# a = [[1,2,3],[4,5,6],[7,8,9]]
# c = []

# a_tensor = tf.convert_to_tensor(a, name = "a", dtype = tf.int32)

# i = tf.train.range_input_producer(3, shuffle=False).dequeue()
# for t in range(a_tensor.get_shape()[0]):
# 	b = a_tensor[t]
# 	c.append(b[i])

# sess = tf.Session()
# # Start input enqueue threads.
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# # Run training steps or whatever
# try:
# 	for step in range(3):
# 		print sess.run(c)
# except Exception,e:
# 	#Report exceptions to the coordinator
# 	coord.request_stop(e)
# coord.request_stop()
# # Terminate as usual.  It is innocuous to request stop twice.
# coord.join(threads)
# sess.close()
'''

# train_data, valid_data, test_data = reader.raw_data("/home/linli/data/BPIC2014")
# batch_len, input_train_event, input_cluster, input_train_attr, target_train_event, target_train_attr =reader.data_clip(train_data, 5, 20, True)

# x,y,c,a,t_a = reader.data_producer(input_train_event, input_cluster, input_train_attr, target_train_event, target_train_attr, batch_len, 20, 5, name =None)

# sess = tf.Session()
# # Start input enqueue threads.
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# # Run training steps or whatever
# try:
# 	for step in range(3):
# 		print sess.run(y)
# except Exception,e:
# 	#Report exceptions to the coordinator
# 	coord.request_stop(e)
# coord.request_stop()
# # Terminate as usual.  It is innocuous to request stop twice.
# coord.join(threads)
# sess.close()
a = tf.constant([[2, 3],[1, 2]])
a_reshape = tf.reshape(a, [4])
b = tf.constant([[[0, 1],[2, 3]], [[1,2],[1,2]]])

b = tf.transpose(b, [1,0,2])
b = tf.reshape(b, [2,-1])
b_ = tf.reshape(b, [2, 2, 2])
b = tf.transpose(b_ ,[1,0,2])
b = tf.reduce_sum(b, 2)
# mul = tf.multiply(a_reshape, b)
with tf.Session() as sess:
	print(sess.run(b))