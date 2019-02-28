import tensorflow as tf
import numpy
import time
import os
import h5py
import readWords2016 as read
import string
import pyxdameraulevenshtein as dl

os.environ['CUDA_VISIBLE_DEVICES']='1'

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("datadir", None, "Directory to save and load midi music files.")
flags.DEFINE_string("traindir", "2016/2016_new", "Directory to save checkpoints and gnuplot files.")
flags.DEFINE_integer("batch_size", 20, "Batch size of training.")
flags.DEFINE_integer("num_unroll_step", 5, "The history length of process.")
flags.DEFINE_integer("num_features", 60, "Num of features of activity.")
flags.DEFINE_integer("num_features_cluster", 20, "Num of features of cluster.")
flags.DEFINE_integer("hidden_size_e0", 100, "Hidden size of lstm nn for e0.")
flags.DEFINE_integer("hidden_size_e1", 100, "Hidden size of lstm nn for e1.")
flags.DEFINE_integer("hidden_size_e2", 100, "Hidden size of lstm nn for e2.")
flags.DEFINE_integer("hidden_size_e3", 50, "Hidden size of lstm nn for e3.")
flags.DEFINE_integer("hidden_size_e4", 50, "Hidden size of lstm nn for e1.")
flags.DEFINE_integer("hidden_size_e5", 50, "Hidden size of lstm nn for e2.")
flags.DEFINE_integer("hidden_size_c", 100, "Hidden size of c .")
flags.DEFINE_integer("num_layer_1", 2, "Number of layers in lstm1.")
flags.DEFINE_integer("num_memory_feature", 100, "Dimensions of memory vector features.")
flags.DEFINE_integer("hidden_size_A", 20, "Hidden size of attention.")
flags.DEFINE_integer("num_layer_MB", 2, "Number of layers in MB.")
flags.DEFINE_integer("hidden_size_cluster", 20, "Hidden size of Cluster Enhance Layer.")
flags.DEFINE_integer("hidden_size_2", 32, "Hidden size of lstm2.")
flags.DEFINE_integer("num_layer_2", 2, "Number of layers in lstm2.")
flags.DEFINE_float("dropout_prob", 0.2, "Dropout probability. 0.0 disables dropout.")
flags.DEFINE_float("max_grad_norm", 5.0,              # 5.0, 10.0
                   "the maximum permissible norm of the gradient.")
flags.DEFINE_float("init_scale", 0.05,                # .1, .04
                   "the initial scale of the weights.")
flags.DEFINE_float("reg_scale", 1.0, "L2 regularization scale.")

flags.DEFINE_integer("num_epochs_full_lr", 15,
                     "The number of epochs with full learning rate.")
flags.DEFINE_integer("num_epochs", 30, "The exact number of epochs.")
flags.DEFINE_float("learning_rate", 0.2, "The initial learning rate.")
flags.DEFINE_float("lr_decay", 0.75, "Learning rate decay.")
flags.DEFINE_float("forget_bias", 0.1, "The bias of forget gate in LSTM cell.")

flags.DEFINE_boolean("shuffle", False, "Shuffle the dataset.")
flags.DEFINE_string("end_of_case", "[EOC]", "The character of the end of the case.")
flags.DEFINE_integer("max_num_batches", 100, "The max number of batches.")
FLAGS = flags.FLAGS

maxSuffixLen = 20 * FLAGS.num_unroll_step


resultFile = open("2016/e1_ResultFile_attention_allevents" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".logging.csv", 'w')

resultFile.write("FLAGS.batch_size, %0.d\n" %(FLAGS.batch_size))
resultFile.write("FLAGS.num_unroll_step, %0.d\n" %(FLAGS.num_unroll_step))
resultFile.write("hiddenSize, e0,e1,c,%0.d,%0.d,%0.d\n" %(FLAGS.hidden_size_e0, FLAGS.hidden_size_e1, FLAGS.hidden_size_cluster))
resultFile.write("droputProb, %.3f\n" %(FLAGS.dropout_prob))
resultFile.write("numLayers, %0.d\n" %(FLAGS.num_layer_1))
resultFile.write("FLAGS.max_grad_norm, %.3f\n" %(FLAGS.max_grad_norm))
resultFile.write("initScale, %.3f\n" %(FLAGS.init_scale))
resultFile.write("numEpochsFullLR, %0.d\n" %(FLAGS.num_epochs_full_lr))
resultFile.write("numEpochs, %0.d\n" %(FLAGS.num_epochs))
resultFile.write("baseLearningRate, %.3f\n" %(FLAGS.learning_rate))
resultFile.write("lrDecay, %.3f\n" %(FLAGS.lr_decay))
resultFile.write("forgetBias, %.3f\n" %(FLAGS.forget_bias))

if (FLAGS.shuffle):
    resultFile.write("Shuffle\n")
else:
    resultFile.write("NoShuffle\n")

resultFile.write("\nDataset, VocabSize, TrainWords, ValidWords, Epoch, TrainPrecision, TrainPerplexity, TrainCrossEntropy, Epoch, ValidPrecision, ValidPerplexity, ValidCrossEntropy\n")
resultFile.flush()
os.fsync(resultFile.fileno())

def linear(inp, output_dim, scope_name=None, stddev=1.0, reuse_scope=False):
  norm = tf.random_normal_initializer(stddev=stddev, dtype=tf.float32)
  const = tf.constant_initializer(0.0, dtype=tf.float32)
  with tf.variable_scope(scope_name or 'G/linear') as scope:
    scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
    if reuse_scope:
      scope.reuse_variables()
    #print('inp.get_shape(): {}'.format(inp.get_shape()))
    w = tf.get_variable('w', [inp.get_shape()[1], output_dim], initializer=norm, dtype=tf.float32)
    b = tf.get_variable('b', [output_dim], initializer=const, dtype=tf.float32)
  return tf.matmul(inp, w) + b

def plus(a,b):
    c = []
    for e1,e2 in zip(a,b):
        c.append(e1+e2)
    return c

path = "../data"

dataset = "BPIC2016_cluster.txt"

input_filename = os.path.join(path, dataset)
target_filename = os.path.join(path, dataset)

# Split the whole dataset into 7:2:1 for training/ validation/ test respectively
input_data_splited, target_data_splited, inputWord2Id, targetWord2Id, inputVocabulary, targetVocabulary, cluster2Id = read.data_split(path, 
    dataset, dataset, True)
# Get the validation words and training words
input_validation_traces, target_validation_traces, input_training_traces, target_training_traces = read.validation_training_data(input_data_splited, 
    target_data_splited, inputWord2Id, targetWord2Id, cluster2Id)
input_test_traces, target_test_traces = read.test_data(input_data_splited, target_data_splited, inputWord2Id, targetWord2Id, cluster2Id)
    

# input_vocabulary is a list of length for each event attr
input_vocabulary = [len(event_attr_dict) for event_attr_dict in inputWord2Id]
target_vocabulary = [len(event_attr_dict) for event_attr_dict in targetWord2Id]

input_event_number = len(input_vocabulary)

for i, (content_input, content_target) in enumerate(zip(input_vocabulary, target_vocabulary)):
    print("Input  Dataset: " + dataset + ", Vocabulary Size of Event attr" + str(i) + ": %.0f\n" % (input_vocabulary[i]))
    print("Target Dataset: " + dataset + ", Vocabulary Size of Event attr" + str(i) + ": %.0f\n" % (target_vocabulary[i]))

input_cluster_number = read._get_cluster_number(input_filename)
print("Input  Dataset: " + dataset + ", Number of Clusters: %d\n" % (input_cluster_number))
    
for i, content in enumerate(cluster2Id):
    print("Input  Dataset: " + dataset + ", Vocabulary Size of Cluster" + str(i) + ": %.0f\n" % (len(cluster2Id[i])))

# Need to rebuild the graph for each dataset because of the different vocabulary sizes
tf.reset_default_graph()

with tf.Graph().as_default() as G:
    # Create placeholders for inputs and targets for an event extraction network
    input_data_c0 = tf.placeholder(tf.int64, [FLAGS.batch_size], name="Input_Cluster0")
    input_data_c1 = tf.placeholder(tf.int64, [FLAGS.batch_size], name="Input_Cluster1")
    
    input_data_es = [tf.placeholder(tf.int64, [FLAGS.batch_size, FLAGS.num_unroll_step], name="Input_Event"+str(i)) for i in range(len(inputWord2Id))]
    
    targets = [tf.placeholder(tf.int64, [FLAGS.batch_size, FLAGS.num_unroll_step], name="Target_Event"+str(i)) for i in range(len(inputWord2Id))]
    
    # Embeddings for each cluster and each event attr respectively, here we need to know how many cluster we have, and how many event attr in trace
    embedding_clusters = [tf.Variable(tf.random_uniform([len(cluster_vocabulary), FLAGS.num_features_cluster], -FLAGS.init_scale, FLAGS.init_scale),  name="Embedding_c" + str(i)) for i, cluster_vocabulary in enumerate(cluster2Id)]
    embedding_events = [tf.Variable(tf.random_uniform([event_vocabulary, FLAGS.num_features], -FLAGS.init_scale, FLAGS.init_scale),  name="Embedding_e" + str(i)) for i, event_vocabulary in enumerate(input_vocabulary)]
    embedding_target_event = [tf.Variable(tf.random_uniform([event_t_vocabulary, FLAGS.num_features], -FLAGS.init_scale, FLAGS.init_scale),  name="Embedding_t" +str(i)) for i,event_t_vocabulary in enumerate(target_vocabulary)]
            
    input_c0 = tf.nn.embedding_lookup(embedding_clusters[0], input_data_c0, name="Embedding_Lookup_c0")
    input_c1 = tf.nn.embedding_lookup(embedding_clusters[1], input_data_c1, name="Embedding_Lookup_c1")
    
    input_es = [tf.nn.embedding_lookup(embedding_events[i], input_data_es[i], name="Embedding_Lookup_e"+str(i)) for i in range(len(inputWord2Id))]
    
    input_ts = [tf.nn.embedding_lookup(embedding_target_event[i], target, name="Embedding_Lookup_t"+str(i)) for i,target in enumerate(targets)]
    
    # Add a probabilistic dropout to inputs as well
    if (FLAGS.dropout_prob > 0):
        input_es = [tf.nn.dropout(input_e, 1.0-FLAGS.dropout_prob, name="Input_Dropout_e"+str(i)) for input_e in input_es]
    # reshape inputs
    input_es_split = [tf.split(input_e, FLAGS.num_unroll_step, 1) for input_e in input_es]
    input_es = []
    for i, input_e_split in enumerate(input_es_split):
        print(i)  
        input_es.append([tf.squeeze(input_, [1]) for input_ in input_e_split])
    
    with tf.variable_scope("lstm_event_e0") as scope:
        # Create the individual LSTM layers
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size_e0, forget_bias=0.0)
        # Add a propabilistic dropout to cells of the LSTM layer
        if (FLAGS.dropout_prob > 0):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0-FLAGS.dropout_prob)
        # Replicate this (including dropout) to additional layers
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * FLAGS.num_layer_1)

        # initial state of all cells is zero
        initialState = cell.zero_state(FLAGS.batch_size, tf.float32)

        outputs_e0 = []
        state = initialState
        for i,input_ in enumerate(input_es[0]):
            if i > 0: scope.reuse_variables()            
            input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_e0,
                                scope_name='input_layer', reuse_scope=(i!=0)))
            
            output, state = cell(input_, state)
            print("ouput.shape,{}".format(output.get_shape()))
            outputs_e0.append(output)

        # connect inputs and outputs to the RNN layers
        print("----------outpute0-------------")
        print(outputs_e0[0].get_shape())
        print("------------------------------------")

    with tf.variable_scope("lstm_event_e1") as scope:
        # Create the individual LSTM layers
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size_e1, forget_bias=0.0)
        # Add a propabilistic dropout to cells of the LSTM layer
        if (FLAGS.dropout_prob > 0):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0-FLAGS.dropout_prob)
        # Replicate this (including dropout) to additional layers
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * FLAGS.num_layer_1)

        # initial state of all cells is zero
        initialState = cell.zero_state(FLAGS.batch_size, tf.float32)

        outputs_e1 = []
        state = initialState
        for i,input_ in enumerate(input_es[1]):
            if i > 0: scope.reuse_variables()            
            input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_e1,
                                scope_name='input_layer', reuse_scope=(i!=0)))
            
            output, state = cell(input_, state)
            print("ouput.shape,{}".format(output.get_shape()))
            outputs_e1.append(output)

        # connect inputs and outputs to the RNN layers
        print("----------outpute1-------------")
        print(outputs_e1[0].get_shape())
        print("------------------------------------")

    with tf.variable_scope("lstm_event_e2") as scope:
        # Create the individual LSTM layers
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size_e2, forget_bias=0.0)
        # Add a propabilistic dropout to cells of the LSTM layer
        if (FLAGS.dropout_prob > 0):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0-FLAGS.dropout_prob)
        # Replicate this (including dropout) to additional layers
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * FLAGS.num_layer_1)

        # initial state of all cells is zero
        initialState = cell.zero_state(FLAGS.batch_size, tf.float32)

        outputs_e2 = []
        state = initialState
        for i,input_ in enumerate(input_es[2]):
            if i > 0: scope.reuse_variables()            
            input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_e2,
                                scope_name='input_layer', reuse_scope=(i!=0)))
            
            output, state = cell(input_, state)
            print("ouput.shape,{}".format(output.get_shape()))
            outputs_e2.append(output)

        # connect inputs and outputs to the RNN layers
        print("----------outpute1-------------")
        print(outputs_e2[0].get_shape())
        print("------------------------------------")

    outputs_z = []
    variable_b = []
    z = tf.concat([tf.multiply(outputs_e0[-1], outputs_e1[-1]),
        tf.multiply(outputs_e0[-1], outputs_e2[-1]),
        outputs_e0[-1],
        outputs_e1[-1],
        outputs_e2[-1]],
        1)
    with tf.variable_scope("modulator_e"):
        z_w = tf.get_variable("z_w", [z.get_shape()[1], 3], dtype=tf.float32)
        z_b = tf.get_variable("z_b", [3], dtype = tf.float32)
        logits_z = tf.nn.xw_plus_b(z, z_w, z_b)
        b = tf.sigmoid(logits_z)
    for i in range(FLAGS.num_unroll_step):
      output_event = tf.expand_dims(outputs_e0[i], -1)
      output_attr_1 = tf.expand_dims(outputs_e1[i], -1)
      output_attr_2 = tf.expand_dims(outputs_e2[i], -1)
      input_z = tf.reshape(tf.transpose(tf.concat([output_event,output_attr_1,output_attr_2], -1), [1,0,2]), 
        [output_event.get_shape()[1], -1])
      output_z = tf.transpose(tf.reshape(tf.multiply(tf.reshape(b, [-1]), input_z), [output_event.get_shape()[1], FLAGS.batch_size, -1]),[1,0,2])
      output_z = tf.reduce_sum(output_z, 2)
      outputs_z.append(output_z)
      variable_b.append(tf.expand_dims(tf.reduce_mean(b,0),0))

    weight_b = tf.reduce_mean(tf.concat(variable_b, 0),0)

    z = tf.concat([tf.multiply(outputs_e0[-1], outputs_e1[-1]),
        tf.multiply(outputs_e1[-1], outputs_e2[-1]),
        outputs_e0[-1],
        outputs_e1[-1],
        outputs_e2[-1]],
        1)
    outputs_z_a1 = []
    variable_b_a1 = []
    with tf.variable_scope("modulator_a2"):
        z_w = tf.get_variable("z_w", [z.get_shape()[1], 3], dtype=tf.float32)
        z_b = tf.get_variable("z_b", [3], dtype = tf.float32)
        logits_z = tf.nn.xw_plus_b(z, z_w, z_b)
        b_a1 = tf.sigmoid(logits_z)
    for i in range(FLAGS.num_unroll_step):
      output_event = tf.expand_dims(outputs_e0[i], -1)
      output_attr_1 = tf.expand_dims(outputs_e1[i], -1)
      output_attr_2 = tf.expand_dims(outputs_e2[i], -1)
      input_z_a1 = tf.reshape(tf.transpose(tf.concat([output_event,output_attr_1,output_attr_2], -1), [1,0,2]), [output_event.get_shape()[1], -1])
      output_z_a1 = tf.transpose(tf.reshape(tf.multiply(tf.reshape(b_a1, [-1]), input_z_a1), [output_event.get_shape()[1], FLAGS.batch_size, -1]),[1,0,2])
      output_z_a1 = tf.reduce_sum(output_z_a1, 2)
      outputs_z_a1.append(output_z_a1)
      variable_b_a1.append(tf.expand_dims(tf.reduce_mean(b_a1,0),0))
    weight_b_a1 = tf.reduce_mean(tf.concat(variable_b_a1, 0),0)
    

    z = tf.concat([tf.multiply(outputs_e0[-1], outputs_e2[-1]),
        tf.multiply(outputs_e1[-1], outputs_e2[-1]),
        outputs_e0[-1],
        outputs_e1[-1],
        outputs_e2[-1]],
        1)
    outputs_z_a2 = []
    variable_b_a2 = []
    with tf.variable_scope("modulator_a1"):
        z_w = tf.get_variable("z_w", [z.get_shape()[1], 3], dtype=tf.float32)
        z_b = tf.get_variable("z_b", [3], dtype = tf.float32)
        logits_z = tf.nn.xw_plus_b(z, z_w, z_b)
        b_a2 = tf.sigmoid(logits_z)
    for i in range(FLAGS.num_unroll_step):
      output_event = tf.expand_dims(outputs_e0[i], -1)
      output_attr_1 = tf.expand_dims(outputs_e1[i], -1)
      output_attr_2 = tf.expand_dims(outputs_e2[i], -1)
      input_z_a2 = tf.reshape(tf.transpose(tf.concat([output_event,output_attr_1,output_attr_2], -1), [1,0,2]), [output_event.get_shape()[1], -1])
      output_z_a2 = tf.transpose(tf.reshape(tf.multiply(tf.reshape(b_a2, [-1]), input_z_a2), [output_event.get_shape()[1], FLAGS.batch_size, -1]),[1,0,2])
      output_z_a2 = tf.reduce_sum(output_z_a2, 2)
      outputs_z_a2.append(output_z_a2)
      variable_b_a2.append(tf.expand_dims(tf.reduce_mean(b_a2,0),0))
    weight_b_a2 = tf.reduce_mean(tf.concat(variable_b_a2, 0),0)
    
    #the prediction layer
    with tf.variable_scope("decoder1") as scope:
        # Create the individual LSTM layers
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size_2, forget_bias=0.0)
        # Add a propabilistic dropout to cells of the LSTM layer
        if (FLAGS.dropout_prob > 0):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0-FLAGS.dropout_prob)
        # Replicate this (including dropout) to additional layers
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * FLAGS.num_layer_2)

        # initial state of all cells is zero
        initialState = cell.zero_state(FLAGS.batch_size, tf.float32)

        state = initialState

        outputs_2 = []
        states_2 = []
        for i,input_ in enumerate(outputs_z):
            #input_ = tf.concat([input_, output_c],1)
            if i > 0: scope.reuse_variables()
            input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_2,
                            scope_name='input_layer', reuse_scope=(i!=0)))
            
            output, state = cell(input_, state)
            outputs_2.append(output)
            states_2.append(state)

    output_e = tf.reshape(tf.concat(outputs_2,1), [-1, FLAGS.hidden_size_2])

    with tf.variable_scope("decoder2") as scope:
        # Create the individual LSTM layers
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size_2, forget_bias=0.0)
        # Add a propabilistic dropout to cells of the LSTM layer
        if (FLAGS.dropout_prob > 0):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0-FLAGS.dropout_prob)
        # Replicate this (including dropout) to additional layers
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * FLAGS.num_layer_2)

        # initial state of all cells is zero
        initialState = cell.zero_state(FLAGS.batch_size, tf.float32)

        state = initialState

        outputs_2 = []
        states_2 = []
        for i,input_ in enumerate(outputs_z_a1):
            #input_ = tf.concat([input_, output_c],1)
            if i > 0: scope.reuse_variables()
            input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_2,
                            scope_name='input_layer', reuse_scope=(i!=0)))
            
            output, state = cell(input_, state)
            outputs_2.append(output)
            states_2.append(state)

    output_a1 = tf.reshape(tf.concat(outputs_2,1), [-1, FLAGS.hidden_size_2])

    with tf.variable_scope("decoder3") as scope:
        # Create the individual LSTM layers
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size_2, forget_bias=0.0)
        # Add a propabilistic dropout to cells of the LSTM layer
        if (FLAGS.dropout_prob > 0):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0-FLAGS.dropout_prob)
        # Replicate this (including dropout) to additional layers
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * FLAGS.num_layer_2)

        # initial state of all cells is zero
        initialState = cell.zero_state(FLAGS.batch_size, tf.float32)

        state = initialState

        outputs_2 = []
        states_2 = []
        for i,input_ in enumerate(outputs_z_a2):
            #input_ = tf.concat([input_, output_c],1)
            if i > 0: scope.reuse_variables()
            input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_2,
                            scope_name='input_layer', reuse_scope=(i!=0)))
            
            output, state = cell(input_, state)
            outputs_2.append(output)
            states_2.append(state)

    output_a2 = tf.reshape(tf.concat(outputs_2,1), [-1, FLAGS.hidden_size_2])

    outputs = []
    states = states_2

    outputs.append(output_e)
    outputs.append(output_a1)
    outputs.append(output_a2)

    target_vocabulary_num = len(target_vocabulary)
    logitses = []
    softmaxes = []
    losses = []
    correct_predictions = []
    prediction_indices = []
    numCorrectPredictions = []
    accuracys = []
    crossEntropys = []

    # Do a softmax to identify the highest probability word
    for i in range(target_vocabulary_num):
        logits = linear(outputs[i], target_vocabulary[i], scope_name='output_logits'+str(i), reuse_scope = False)
        logitses.append(logits)
        softmax = tf.nn.softmax(logits, name="softmax"+str(i))
        softmaxes.append(softmax)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example( [logits], 
            [tf.reshape(targets[i], [-1])], weights=[tf.ones([FLAGS.batch_size * FLAGS.num_unroll_step])], name="SeqLossByExample"+str(i))
        losses.append(loss)
        # Compute the accuracy
        correct_prediction = tf.cast(tf.nn.in_top_k(logits, tf.reshape(targets[i], [-1]), 1), tf.float32)
        correct_predictions.append(correct_prediction)
        _ , prediction_indice = tf.nn.top_k(softmax, 1)
        prediction_indices.append(tf.reshape(prediction_indice, [FLAGS.batch_size, -1]))
        numCorrectPrediction = tf.reduce_sum(correct_prediction, name="SumCorrectPredictions"+str(i))
        numCorrectPredictions.append(numCorrectPrediction)
        accuracy = tf.reduce_mean(correct_prediction, name="MeanCorrectPredictions"+str(i))
         # Compute the cross entropy
        oneHottargets = tf.one_hot(targets[i], depth=target_vocabulary[i], on_value=1.0, off_value=0.0)
        reshapedtargets = tf.reshape(oneHottargets, shape=(FLAGS.batch_size*FLAGS.num_unroll_step, target_vocabulary[i]))
        crossEntropy = tf.reduce_mean(-tf.reduce_sum(reshapedtargets * tf.log(tf.sigmoid(logits)), reduction_indices=[1]), reduction_indices=[0], name="MeanCrossEntropy"+str(i))
        crossEntropys.append(crossEntropy)


    print("----------logits0-------------")
    print(logits[0].get_shape())
    print("------------------------------------")    
    

    cost = tf.reduce_sum(losses[0] + losses[1] + losses[2], name="SumLoss") / FLAGS.batch_size
    finalState = states[FLAGS.num_unroll_step-1]

    # learning rate is a variable, but not trainable
    learningRate = tf.Variable(0.0, trainable=False, name="LearningRate")
    trainableVars = tf.trainable_variables()
    # compute gradients of all trainable variables w.r.t cost, then clip the gradients, prevent from getting too large too fast
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainableVars), FLAGS.max_grad_norm)
    # Define the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    # and tell it to work on the gradients for the trainable variables
    train_op = optimizer.apply_gradients(zip(grads, trainableVars))

    saver = tf.train.Saver()

    #the total number of batches for one epoch
    epochNumBatches = len([1 for _ in read.words_iterator(input_training_traces, target_training_traces, FLAGS.batch_size, FLAGS.num_unroll_step)])
    print("NumBatches: " + str(epochNumBatches))
    # Initialize all variables in the computational graph
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    # Training starts here
    for i in range(FLAGS.num_epochs):
        # Adjust the learning rate by decaying it
        # and set the appropriate variable in the graph
        sess.run(tf.assign(learningRate, FLAGS.learning_rate*FLAGS.lr_decay**max(i - FLAGS.num_epochs_full_lr, 0.0)))
        # Get the learning rate and print it
        print("Dataset: [" + dataset + "/" + dataset + "] Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(learningRate)))
        start_time = time.time()
        # accumulated cross entropy
        accumCrossEnt = [0.0 for _ in range(target_vocabulary_num)]
        # accumulated costs over the unroll steps
        accumCosts = 0.0
        # number of correct predictions
        accumNumCorrPred = [0.0 for _ in range(target_vocabulary_num)]
        # number of iterations/unroll steps
        iters = 0.0
        for batchNum, (c, x, y) in enumerate(
            read.words_iterator(input_training_traces, target_training_traces,FLAGS.batch_size, FLAGS.num_unroll_step)):
            batchNumCorrPred, batchCrossEnt, batchCost, state, _ = sess.run([numCorrectPredictions, 
                crossEntropys, cost, finalState, train_op], {input_data_c0: c[0], input_data_c1: c[1],
                input_data_es[0]: x[0], input_data_es[1]: x[1], input_data_es[2]: x[2], 
                targets[0]:y[0], targets[1]:y[1], targets[2]:y[2]})
            
            accumCosts += batchCost
            accumNumCorrPred = plus(accumNumCorrPred, batchNumCorrPred)
            accumCrossEnt = plus(accumCrossEnt, batchCrossEnt)
            iters += FLAGS.num_unroll_step
                
            if (epochNumBatches > 10):
                if batchNum % (epochNumBatches // 10) == 10:
                    print("Dataset: [" + dataset + "/" + dataset +
                        "] Epoch percent: %.3f perplexity: %.3f speed: %.0f wps: " %
                        (batchNum * 1.0 / epochNumBatches, numpy.exp(accumCosts / iters), iters * FLAGS.batch_size / (time.time() - start_time)))
                    print("number of correct predictions: [%s]" %(' '.join([str(number) for number in accumNumCorrPred])))
                    print("precision: [%s] " %(' '.join([str(number / (iters * FLAGS.batch_size)) for number in accumNumCorrPred])))
                    print("cross-entropy: [%s]" %(' '.join([str(number / batchNum) for number in accumCrossEnt])))  

        thisPrecision = [float(number) / float(iters * FLAGS.batch_size) for number in accumNumCorrPred]
        thisPerplexity = numpy.exp(accumCosts / iters)
        thisCrossEntropy = [number / batchNum for number in accumCrossEnt]
        print("Dataset: [" + dataset + "/" + dataset + "] Epoch summary-- perplexity: %.3f speed: %.0f " %
                  (numpy.exp(accumCosts / iters), iters * FLAGS.batch_size / (time.time() - start_time)))
        print("number of correct predictions: [%s]" %(' '.join([str(number) for number in accumNumCorrPred])))
        print("precision: [%s]" %(' '.join([str(number) for number in thisPrecision])))
        print("cross-entropy: [%s]" %(' '.join([str(number) for number in thisCrossEntropy])))
            
        resultFile.write("[" + dataset + "/" + dataset + "], %d, %d, %d, %d precision: [%s] perplexity: [%.3f] cross-entropy: [%s] \n"% (inputVocabulary, 
            len(input_training_traces), len(input_validation_traces), i,' '.join([str(number) for number in thisPrecision]) ,thisPerplexity ,
                ' '.join([str(number) for number in thisCrossEntropy])))
        resultFile.flush()
        os.fsync(resultFile.fileno())

        checkpoint_path = os.path.join(FLAGS.traindir, 'model.ckpt')

        if i%10 == 0:
            saver.save(sess, checkpoint_path, global_step = i)
        
    # Validation starts here. Differences are the use of validwords for data and tf.no_op() instead of the train_op() built into the graph
    validateNumBatches = len([1 for _ in read.words_iterator(input_validation_traces, target_validation_traces, FLAGS.batch_size, FLAGS.num_unroll_step)])
    start_time = time.time()
    # accumulated cross entropy
    accumCrossEnt = [0.0 for _ in range(target_vocabulary_num)]
    # accumulated costs over the unroll steps
    accumCosts = 0.0
    # number of correct predictions
    accumNumCorrPred = [0.0 for _ in range(target_vocabulary_num)]
    # number of iterations/unroll steps
    iters = 0
    for batchNum, (c, x, y) in enumerate(read.words_iterator(input_validation_traces, target_validation_traces, FLAGS.batch_size, FLAGS.num_unroll_step)):
        batchNumCorrPred, batchCrossEnt, batchCost, state, _ = sess.run([numCorrectPredictions, crossEntropys, cost, finalState, tf.no_op()], {
            input_data_c0: c[0], input_data_c1: c[1],input_data_es[0]: x[0], input_data_es[1]: x[1], input_data_es[2]: x[2],
            targets[0]:y[0], targets[1]:y[1], targets[2]:y[2]})
        accumCosts += batchCost
        accumNumCorrPred = plus(accumNumCorrPred, batchNumCorrPred)
        accumCrossEnt = plus(accumCrossEnt, batchCrossEnt)
        iters += FLAGS.num_unroll_step
        if (validateNumBatches > 10):
            if batchNum % (validateNumBatches // 10) == 10:
                print("Dataset: [" + dataset + "/" + dataset +"] Epoch percent: %.3f perplexity: %.3f speed: %.0f wps: " %
                    (batchNum * 1.0 / validateNumBatches, numpy.exp(accumCosts / iters), iters * FLAGS.batch_size / (time.time() - start_time)))
                print("number of correct predictions: " + ' '.join([str(number) for number in accumNumCorrPred]))
                print("precision: " + ' '.join([str(number / (iters * FLAGS.batch_size)) for number in accumNumCorrPred]))
                print("cross-entropy: " + ' '.join([str(number / batchNum) for number in accumCrossEnt]))   

    thisPrecision = [float(number) / float(iters * FLAGS.batch_size) for number in accumNumCorrPred]
    thisPerplexity = numpy.exp(accumCosts / iters)
    thisCrossEntropy = [number / batchNum for number in accumCrossEnt]
    print("Dataset: [" + dataset + "/" + dataset + "] Epoch summary-- perplexity: %.3f speed: %.0f " %
        (numpy.exp(accumCosts / iters), iters * FLAGS.batch_size / (time.time() - start_time),))
    print("number of correct predictions: " + ' '.join([str(number) for number in accumNumCorrPred]))
    print("precision: " + ' '.join([str(number) for number in thisPrecision]))
    print("cross-entropy: " + ' '.join([str(number) for number in thisCrossEntropy]))
            
    resultFile.write("[" + dataset + "/" + dataset + "], %d, %d, %d, %d precision: [%s] perplexity: [%.3f] cross-entropy: [%s] \n"% (inputVocabulary, 
            len(input_training_traces), len(input_validation_traces), i,' '.join([str(number) for number in thisPrecision]) ,thisPerplexity ,
                ' '.join([str(number) for number in thisCrossEntropy])))
    resultFile.flush()
    os.fsync(resultFile.fileno())

    # Invert the Word dictionary
    inputId2Word = {v: k for k, v in inputWord2Id[0].iteritems()}
    if (targetVocabulary==inputVocabulary):
        outputFile = open(
            "2016/new_SuffixPrediction{0}Dataset{1}FLAGS.batch_size{2}unrollSteps{3}.txt".format(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), dataset, FLAGS.batch_size, FLAGS.num_unroll_step), "w")
        outputFile.write("NormedDLDistance\n")
        print("Suffix Prediction ...\n")
        # Suffix prediction starts here,
        endOfCaseId = inputWord2Id[0]['[EOC]']
        # An zero array for the targets0 (which we don't need)
        y = numpy.zeros((FLAGS.batch_size, FLAGS.num_unroll_step), dtype=numpy.int32)
        # need to know how many batches we're sending through.
        test_sentences = read.read_sentences(input_test_traces, FLAGS.num_unroll_step)
        test_epochNumBatches = int(len(test_sentences)/FLAGS.batch_size)
        print("Test NumBatches: " + str(test_epochNumBatches))
            
        numBatches = min(test_epochNumBatches, FLAGS.max_num_batches)
        sum_distance = 0.0
        
        for batchNum in range(numBatches):
            clusters, prefixes, suffixes = read.read_batch_of_sentences(test_sentences, FLAGS.batch_size, FLAGS.num_unroll_step, batchNum * FLAGS.batch_size)
            print("eocID: " + str(endOfCaseId))
            print("suffix_end: " +str(suffixes[0][0][-1])) 
            c = clusters  
            x = prefixes
            h = numpy.empty((FLAGS.batch_size, maxSuffixLen), dtype=numpy.int32)
            for i in range(maxSuffixLen):
                batchPredictions, batchsoftmaxes, batchFinalState, _ = sess.run([prediction_indices, softmaxes, finalState, tf.no_op()], {
                    input_data_c0: c[0], input_data_c1: c[1],
                    input_data_es[0]: x[0], input_data_es[1]: x[1], targets[0]:y, targets[1]:y})

                predVal = numpy.array(batchPredictions, dtype = numpy.int32)
                # print("predVal_shape: ")
                # print(predVal.shape)
                
                for j in range(FLAGS.batch_size):
                    h[j][i] = predVal[0][j][-1]
                x = predVal
            suffixes_predicted = numpy.empty((FLAGS.batch_size), dtype='object')
            for j in range(FLAGS.batch_size):
                try:
                    eocIndex = h[j].tolist().index(endOfCaseId) + 1
                except ValueError, e:
                    eocIndex = None
                suffixes_predicted[j] = h[j].tolist()[:eocIndex]
            suffixes_predicted_alpha = suffixes_predicted
            suffixes_alpha = [ [s for s in suffix ] for suffix in suffixes[0]]
            for j in range(FLAGS.batch_size):
                outputFile.write("{0}\n".format(suffixes_predicted_alpha[j]))
                outputFile.write("{0}\n".format(suffixes_alpha[j]))
                if len(suffixes_alpha[j]) > 100:
                    suffixes_alpha_this = suffixes_alpha[j][0:100]
                else:
                    suffixes_alpha_this = suffixes_alpha[j]
                distance = dl.normalized_damerau_levenshtein_distance(suffixes_predicted_alpha[j], suffixes_alpha_this)
                outputFile.write("{0}\n".format(distance))
                sum_distance += distance
            outputFile.flush()
            os.fsync(outputFile.fileno())
            print("Batch {} of {} ".format(batchNum, numBatches))
                
        outputFile.write("average edit_distance: {0}\n".format(sum_distance / (FLAGS.batch_size * numBatches)))
        outputFile.close()
        
    resultFile.close()

