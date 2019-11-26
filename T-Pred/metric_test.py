import tensorflow as tf
import numpy as np

y_true = np.array([[2,1], [1,0], [0,3], [3,2], [0,2]]).astype(np.int64)
y_true = tf.identity(y_true)

y_pred = np.array([[[0.1, 0.2, 0.6, 0.1],[0.1, 0.2, 0.6, 0.1]],
                   [[0.8, 0.05, 0.1, 0.05], [0.8, 0.05, 0.1, 0.05]],
                   [[0.3, 0.4, 0.1, 0.2], [0.3, 0.4, 0.1, 0.2]],
                   [[0.6, 0.25, 0.1, 0.05], [0.6, 0.25, 0.1, 0.05]],
                   [[0.1, 0.2, 0.6, 0.1], [0.1, 0.2, 0.6, 0.1]]
                   ]).astype(np.float32)
y_pred = tf.identity(y_pred)

ap, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 3, name="a_precision_at_k")
running_a_precision_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="a_precision_at_k")

precision, m_precision = tf.metrics.precision_at_k(y_true, y_pred, 3, name='precision')
running_precision_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision")

recall, m_recall = tf.metrics.recall_at_k(y_true, y_pred, 3, name="recall")
running_recall_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall")

sess = tf.Session()

sess.run(tf.local_variables_initializer())
tf_p, p = sess.run([m_precision, precision])
print("TF_P",tf_p)
print('p', p)
print("STREAM_VARS",(sess.run(running_precision_vars)))

# tmp_rank = tf.nn.top_k(y_pred,3)
# print("TMP_RANK",sess.run(tmp_rank))

sess.run(tf.variables_initializer(var_list=running_precision_vars))
tf_p, p = sess.run([m_precision, precision])
print("TF_P",tf_p)
print('p', p)

print("STREAM_VARS",(sess.run(running_precision_vars)))

# tmp_rank = tf.nn.top_k(y_pred,3)
# print("TMP_RANK",sess.run(tmp_rank))

sess.run(tf.variables_initializer(var_list=running_a_precision_vars))
tf_map = sess.run(m_ap)
ap = sess.run(ap)
print("\nTF_MAP",tf_map)
print('ap', ap)
print("STREAM_VARS",(sess.run(running_a_precision_vars)))

print("\nSTREAM_VARS",(sess.run(running_recall_vars)))
sess.run(tf.variables_initializer(var_list=running_recall_vars))
tf_recall = sess.run(m_recall)
recall = sess.run(recall)
print("TF_RECALL",tf_recall)
print('recall', recall)
print("STREAM_VARS",(sess.run(running_recall_vars)))