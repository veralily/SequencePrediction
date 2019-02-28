import tensorflow as tf
checkpoint_dir = "./model"
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
	# Load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		# Get the placeholders from the graph by name
		b = graph.get_operation_by_name("Train/Model/modulator/b").outputs[0]
		print sess.run(b)