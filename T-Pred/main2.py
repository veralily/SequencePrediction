import os
import argparse
import read_data
import logging
import datetime
import time
import Model
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICE'] = '4'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--train_iter', required=True, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr_e', default=0.001, type=float)
parser.add_argument('--lr_t', default=0.001, type=float)
parser.add_argument('--lr_j', default=0.001, type=float)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--gamma', default=1.0, type=float)
parser.add_argument('--T', default=5, type=int)
parser.add_argument('--len', default=5, type=int)
parser.add_argument('--num_gen_t', default=3, type=int)
parser.add_argument('--hidden_size', default=200, type=int)
parser.add_argument('--num_block', default=2, type=int)
parser.add_argument('--res_rate', default=0.2, type=float)
parser.add_argument('--num_head', default=2, type=int)
parser.add_argument('--drop_rate', default=0.2, type=int)
parser.add_argument('--vocab_size', default=10000, type=int)
parser.add_argument('--metric_k',default=10, type=int)

args = parser.parse_args()

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.tst'), 'w') as f:
        f.write('\n'.join([str(k)+','+str(v) for k,v in sorted(vars(args).items(), key=lambda x:x[0])]))
    f.close()


'''remember to change the vocab size '''
path = './T-pred-Dataset'
event_file = os.path.join(path,args.dataset + '_event.txt')
time_file = os.path.join(path,args.dataset + '_time.txt')

FORMAT = "%(asctime)s - [line:%(lineno)s - %(funcName)10s() ] %(message)s"
DATA_TYPE = event_file.split('/')[-1].split('.')[0]
logging.basicConfig(filename='log/{}-{}-{}.log'.format('MM-CPred',
                                                       DATA_TYPE,
                                                       str(datetime.datetime.now())),
                    level=logging.INFO,
                    format=FORMAT)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(FORMAT))
logging.getLogger().addHandler(handler)
logging.info('Start {}'.format(DATA_TYPE))

# Read data
train_data, valid_data, test_data = read_data.data_split(event_file, time_file, shuffle=True)

# initialize the model
model = Model.MM_CPred_2(args)

# Get Config and Create a Session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
sess.run(tf.initialize_all_variables())

# Run Train Epoches
train_i_e, train_t_e, train_i_t, train_t_t = read_data.data_iterator(train_data, args.T, args.len, overlap=True)
gap = 6

alpha = args.alpha
gamma = args.gamma

for epoch in range(args.train_iter):
    logging.info('Training epoch: {}'.format(epoch))
    t0 = time.time()
    i = 0
    sess.run(model.running_MAE_vars_initializer)
    sess.run(model.running_precision_vars_initializer)
    sess.run(model.running_recall_vars_initializer)
    for e_x, e_y, t_x, t_y in read_data.generate_batch(args.batch_size, train_i_e, train_t_e, train_i_t, train_t_t):
        sample_t = read_data.generate_sample_t(args.batch_size, train_i_t, train_t_t)
        i += 1
        feed_dict = {
            model.alpha: alpha,
            model.gamma: gamma,
            model.inputs_e: e_x,
            model.inputs_t: np.maximum(np.log(np.maximum(t_x, 1e-4)), 0),
            model.target_t: t_y,
            model.target_e: e_y,
            model.sample_t: np.maximum(np.log(np.maximum(sample_t, 1e-4)), 0)}

        # Jointly train
        # If it's discriminator iter
        if i % gap == 0:
            _, _ = sess.run([model.train_disc_op, model.train_w_clip_op], feed_dict=feed_dict)
        else:
            _, _, _, _, logits_e, logits_t, MAE, precision, recall = sess.run([model.train_gen_op,
                                                                               model.MAE_op,
                                                                               model.precision_op,
                                                                               model.recall_op,
                                                                               model.logits_e,
                                                                               model.logits_t,
                                                                               model.MAE,
                                                                               model.precision,
                                                                               model.recall
                                                                               ], feed_dict=feed_dict)

        # Train event predictor
        _ = sess.run(model.train_MLE_op, feed_dict=feed_dict)

        if i % 100 == 0:
            logging.info('Training -- Batch:{}  MAE: {}, precision@k: {}, recall@k: {}'.format(i,
                                                                                               MAE,
                                                                                               precision,
                                                                                               recall))
    t1 = time.time()
    logging.info('Training metrics of Epoch: {} Time: {} MAE: {}, precision@k: {}, recall@k: {}'
                 .format(epoch, t1 - t0, MAE, precision, recall))

    # Run validation to update alpha and gamma
    t0 = time.time()
    i = 0
    sess.run(model.running_MAE_vars_initializer)
    sess.run(model.running_precision_vars_initializer)
    sess.run(model.running_recall_vars_initializer)

    valid_i_e, valid_t_e, valid_i_t, valid_t_t = read_data.data_iterator(valid_data, args.T, args.len)

    cross_entropy_sum = 0.0
    huber_loss_sum = 0.0
    gen_loss_sum = 0.0

    for e_x, e_y, t_x, t_y in read_data.generate_batch(args.batch_size, valid_i_e, valid_t_e, valid_i_t, valid_t_t):
        sample_t = read_data.generate_sample_t(args.batch_size, valid_i_t, valid_t_t)
        i += 1
        feed_dict = {
            model.alpha: alpha,
            model.gamma: gamma,
            model.inputs_e: e_x,
            model.inputs_t: np.maximum(np.log(np.maximum(t_x, 1e-4)), 0),
            model.target_t: t_y,
            model.target_e: e_y,
            model.sample_t: np.maximum(np.log(np.maximum(sample_t, 1e-4)), 0)}

        cross_entropy, huber_loss, gen_t_loss, disc_t_loss, _, _, _, MAE, precision, recall = sess.run([
            model.cross_entropy_loss,
            model.huber_loss,
            model.gen_t_loss,
            model.disc_t_loss,
            model.MAE_op,
            model.precision_op,
            model.recall_op,
            model.MAE,
            model.precision,
            model.recall],
            feed_dict=feed_dict)
        cross_entropy_sum += cross_entropy
        huber_loss_sum += huber_loss
        gen_loss_sum += gen_t_loss

    alpha = gen_loss_sum / cross_entropy_sum
    gamma = gen_loss_sum / huber_loss_sum
    t1 = time.time()
    logging.info('Validate Time: {} MAE: {}, precision@k: {}, recall@k: {}'.format(t1-t0,
                                                                                   MAE,
                                                                                   precision,
                                                                                   recall))


# Test
t0 = time.time()
sess.run(model.running_MAE_vars_initializer)
sess.run(model.running_precision_vars_initializer)
sess.run(model.running_recall_vars_initializer)

i_e, t_e, i_t, t_t = read_data.data_iterator(test_data, args.T, args.len)

i = 0
batch_num = len(list(read_data.generate_batch(args.batch_size, i_e, t_e, i_t, t_t)))
logging.info('Evaluation batch num {}'.format(batch_num))
for e_x, e_y, t_x, t_y in read_data.generate_batch(args.batch_size, i_e, t_e, i_t, t_t):
    i += 1
    sample_t = read_data.generate_sample_t(args.batch_size, i_t, t_t)
    feed_dict = {
        model.alpha: alpha,
        model.gamma: gamma,
        model.inputs_e: e_x,
        model.inputs_t: np.maximum(np.log(np.maximum(t_x, 1e-4)), 0),
        model.target_t: t_y,
        model.target_e: e_y,
        model.sample_t: np.maximum(np.log(np.maximum(sample_t, 1e-4)), 0)}

    _, _, _, MAE, precision, recall = sess.run([model.MAE_op,
                                                model.precision_op,
                                                model.recall_op,
                                                model.MAE,
                                                model.precision,
                                                model.recall],
                                               feed_dict=feed_dict)


t1 = time.time()
logging.info('Test Time: {} MAE: {}, precision@k: {}, recall@k: {}'.format(t1-t0,
                                                                           MAE,
                                                                           precision,
                                                                           recall))
