class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.01
  max_grad_norm = 5
  num_layers = 2
  num_steps = 5
  num_gen = 3
  output_length = 5
  hidden_size = 200
  g_size = 200
  filter_output_dim = 200
  filter_size = 3
  keep_prob = 1.0
  res_rate = 0.3
  lr_decay = 0.5
  batch_size = 100
  vocab_size = 52740
  label_size = 10
  LAMBDA = 10
  gamma = 1000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 122911
  label_size = 20
  rnn_mode = 'block'


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  label_size = 20
  rnn_mode = 'block'


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  label_size = 20
  rnn_mode = 'block'
