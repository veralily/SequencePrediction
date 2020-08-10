import os
import argparse
import read_data
import logging
import datetime
import time
import Model
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--train_iter', required=True)
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

# Get Config and Create a Session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

model = Model.MM_CPred(args)
sess.run(tf.initialize_all_variables())

# Define Metrics
MAE = tf.keras.metrics.MeanAbsoluteError()
precision_at_k = tf.keras.metrics.Precision()
recall_at_k = tf.keras.metrics.Recall()

# Run Train Epoches
i_e, t_e, i_t, t_t = read_data.data_iterator(train_data, args.T, args.len)

i = 0
gap = 6

alpha = args.alpha
gamma = args.gamma

for epoch in args.train_iter:
    t0 = time.time()
    MAE.reset_states()
    precision_at_k.reset_states()
    recall_at_k.reset_states()
    for e_x, e_y, t_x, t_y in read_data.generate_batch(args.batch_size, i_e, t_e, i_t, t_t):
        sample_t = read_data.generate_sample_t(args.batch_size, i_t, t_t)
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
            _, logits_e, logits_t = sess.run([model.train_gen_op, model.logits_e, model.logits_t], feed_dict=feed_dict)
            MAE.update_state(y_true=model.target_t,
                             y_pred=logits_t)
            precision_at_k.update_state(y_true=tf.one_hot(model.target_e, depth=args.vocab_size),
                                        y_pred=logits_e)
            recall_at_k.update_state(y_true=tf.one_hot(model.target_e, depth=args.vocab_size),
                                     y_pred=logits_e)
            if i // 100 == 0:
                logging.info('Training metrics Batch partion:{}  MAE: {}, precision@k: {}, recall@k: {}'
                             .format(i,
                                     MAE.result().numpy(),
                                     precision_at_k.result().numpy(),
                                     recall_at_k.result().numpy()))
        # Train event predictor
        _ = sess.run(model.train_event_op, feed_dict=feed_dict)

        # Train time predictor
        _ = sess.run(model.train_time_op, feed_dict=feed_dict)

    t1 = time.time()
    logging.info('Training metrics Epoch: {} Time: {} MAE: {}, precision@k: {}, recall@k: {}'
                 .format(epoch,
                         t1 - t0,
                         MAE.result().numpy(),
                         precision_at_k.result().numpy(),
                         recall_at_k.result().numpoy()))

    # Run validation to update alpha and gamma
    t0 = time.time()
    MAE.reset_states()
    precision_at_k.reset_states()
    recall_at_k.reset_states()

    i_e, t_e, i_t, t_t = read_data.data_iterator(valid_data, args.T, args.len)

    cross_entropy_sum = 0.0
    huber_loss_sum = 0.0
    gen_loss_sum = 0.0
    i = 0
    for e_x, e_y, t_x, t_y in read_data.generate_batch(args.batch_size, i_e, t_e, i_t, t_t):
        sample_t = read_data.generate_sample_t(args.batch_size, i_t, t_t)
        i += 1
        feed_dict = {
            model.alpha: alpha,
            model.gamma: gamma,
            model.inputs_e: e_x,
            model.inputs_t: np.maximum(np.log(np.maximum(t_x, 1e-4)), 0),
            model.target_t: t_y,
            model.target_e: e_y,
            model.sample_t: np.maximum(np.log(np.maximum(sample_t, 1e-4)), 0)}

        cross_entropy, huber_loss, gen_t_loss, disc_t_loss, logits_e, logits_t = sess.run([model.cross_entropy_loss,
                                                                   model.huber_loss,
                                                                   model.gen_t_loss,
                                                                   model.disc_t_loss,model.logits_e,model.logits_t],
                                                                  feed_dict=feed_dict)
        MAE.update_state(y_true=model.target_t,y_pred=logits_t)
        precision_at_k.update_state(y_true=model.target_e, y_pred=logits_e)
        recall_at_k.update_state(y_true=model.target_e, y_pred=logits_e)
        cross_entropy_sum += cross_entropy
        huber_loss_sum += huber_loss
        gen_loss_sum += gen_t_loss

    alpha = gen_loss_sum / cross_entropy_sum
    gamma = gen_loss_sum / huber_loss_sum
    t1 = time.time()
    logging.info('Validate Time: {} MAE: {}, precision@k: {}, recall@k: {}'.format(t1-t0,
                                                                                   MAE.result().numpy(),
                                                                                   precision_at_k.result().numpy(),
                                                                                   recall_at_k.result().numpy()))


# Test
t0 = time.time()
MAE.reset_states()
precision_at_k.reset_states()
recall_at_k.reset_states()

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

    logits_e, logits_t = sess.run([model.cross_entropy_loss,
                                   model.huber_loss,
                                   model.gen_t_loss,
                                   model.disc_t_loss,
                                   model.logits_e,
                                   model.logits_t],
                                  feed_dict=feed_dict)
    MAE.update_state(y_true=model.target_t, y_pred=logits_t)
    precision_at_k.update_state(y_true=model.target_e, y_pred=logits_e)
    recall_at_k.update_state(y_true=model.target_e, y_pred=logits_e)

t1 = time.time()
logging.info('Test Time: {} MAE: {}, precision@k: {}, recall@k: {}'.format(t1-t0,
                                                                           MAE.result(),
                                                                           precision_at_k.result(),
                                                                           recall_at_k.result()))
