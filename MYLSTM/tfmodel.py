import tensorflow as tf
from tensorflow.contrib.slim import conv2d, max_pool2d, fully_connected


def data_type():
    return tf.float32


class model_base:
    def __init__(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.initialized = False

    def inference(self):
        with self.graph.as_default():
            initializer = tf.random_uniform_initializer(-self.config.weight_scale, self.config.weight_scale)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                self.add_placeholder()
                self.add_loss()
                self.add_trainop()
                tf.summary.scalar('loss', self.loss)  # 命名和赋值
                self.merged = tf.summary.merge_all()
                self.writer = tf.summary.FileWriter('tensorboard_log', self.graph)
                # 选定可视化存储目录

                tf.global_variables_initializer().run(session=self.session)
                self.initialized = True

    def add_placeholder(self):
        raise NotImplemented('')

    def add_loss(self):
        raise NotImplemented('')

    def add_trainop(self):
        raise NotImplemented('')

    def fit(self, X, y):
        if not self.initialized:
            self.inference()
        with self.graph.as_default():
            self._fit(X, y)

    def _fit(self, X, y):
        raise NotImplemented('')

    def predict(self, X):
        if not self.initialized:
            self.inference()
        with self.graph.as_default():
            return self._predict(X)

    def _predict(self, X):
        raise NotImplemented('')


class OLS(model_base):
    def __init__(self, config):
        super(OLS, self).__init__()
        self.config = config

    def add_placeholder(self):
        self.input = tf.placeholder(shape=[self.config.batch_size, self.config.input_dimension], dtype=tf.float32)
        self.label = tf.placeholder(shape=[self.config.batch_size], dtype=tf.float32)
        self.predict_input = tf.placeholder(shape=[None, self.config.input_dimension], dtype=tf.float32)

    def add_loss(self):
        W = tf.get_variable('W', shape=(self.config.input_dimension, 1))
        B = tf.get_variable('b', shape=(1,))
        score = tf.reshape(tf.matmul(self.input, W) + B, shape=(self.config.batch_size,))
        self.loss = tf.reduce_sum(tf.square(self.label - score))
        self.score = score
        self.predict_op = tf.reshape(tf.matmul(self.predict_input, W) + B, shape=(-1,))

    def add_trainop(self):
        self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

    def _fit(self, X, y):
        feed_dict = {self.input: X, self.label: y}
        for i in range(1000):
            loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)
            print(loss)

    def _predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.predict_input: X})


class LSTM(model_base):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config

    def add_placeholder(self):
        """add placeholder and onehot encode"""
        with tf.name_scope('placeholder'):
            self.input = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.config.num_of_stages,
                                                           self.config.input_dimension])
            self.label_ = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.num_of_stages])
            self.label = tf.one_hot(self.label_, depth=self.config.num_of_classes)
            self.predict_input = tf.placeholder(shape=[None, self.config.num_of_stages, self.config.input_dimension],
                                                dtype=tf.float32)

    def add_loss(self):
        with tf.name_scope('convolution'):
            pre_cov1 = tf.reshape(self.input, shape=[self.config.batch_size * self.config.num_of_stages,
                                                     self.config.input_dimension])
            pre_cov2 = tf.reshape(pre_cov1,
                                  shape=[self.config.batch_size * self.config.num_of_stages,
                                         self.config.conv1_dimension, self.config.conv2_dimension,
                                         self.config.conv3_dimension])

            after_cov1 = conv2d(pre_cov2, 10, [3, 3], activation_fn=tf.nn.relu)
            after_cov2 = conv2d(after_cov1, 5, [3, 3], activation_fn=tf.nn.relu)
            after_cov3 = max_pool2d(after_cov2, [2, 2])
            after_conv4 = tf.reshape(after_cov3, (self.config.batch_size * self.config.num_of_stages, -1))
            after_fully_connect = fully_connected(after_conv4, self.config.lstm_input_dimension)
            after_fully_connect_reshape = tf.reshape(after_fully_connect, (
            self.config.batch_size, self.config.num_of_stages, self.config.lstm_input_dimension))


            cell, _initial_state = self.add_lstm_cell(self.config.num_of_cells, self.config.keep_prob)
            outputs = []
            state = _initial_state
            # inputs = after_fully_connect_reshape
            inputs=self.input

        with tf.variable_scope("RNN_train"):
            for time_step in range(self.config.num_of_stages):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # cell_out: [batch, hidden_size]
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)  # output: shape[num_steps][batch,hidden_size]
        with tf.name_scope('out_put'):
            output = tf.concat(outputs, axis=1)
            output = tf.reshape(output, [-1, self.config.num_of_hidden_states])
            softmax_w = tf.get_variable(
                "softmax_w", [self.config.num_of_hidden_states, self.config.num_of_classes], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [self.config.num_of_classes], dtype=data_type())
            # [batch*numsteps, vocab_size] 从隐藏语义转化成完全表示
            logits = tf.matmul(output, softmax_w) + softmax_b

            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],  # output [batch*numsteps, vocab_size]
                [tf.reshape(self.label_, [-1])],  # target, [batch_size, num_steps] 然后展开成一维【列表】
                [tf.ones([self.config.batch_size * self.config.num_of_stages], dtype=data_type())])  # weight

            self.loss = tf.reduce_sum(loss) / self.config.batch_size  # 计算得到平均每批batch的误差
            self.predict = tf.reshape(tf.argmax(logits, axis=1),
                                      shape=[self.config.batch_size, self.config.num_of_stages])

    def add_trainop(self):
        with tf.name_scope('train_op'):
            self._lr = tf.Variable(self.config.learning_rate, trainable=False)
            tvars = tf.trainable_variables()
            # clip_by_global_norm: 梯度衰减，具体算法为t_list[i] * clip_norm / max(global_norm, clip_norm)
            # 这里gradients求导，ys和xs都是张量
            # 返回一个长为len(xs)的张量，其中的每个元素都是\grad{\frac{dy}{dx}}
            # clip_by_global_norm 用于控制梯度膨胀,前两个参数t_list, global_norm, 则
            # t_list[i] * clip_norm / max(global_norm, clip_norm)
            # 其中 global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              self.config.max_grad_norm)

            # 梯度下降优化，指定学习速率
            # optimizer = tf.train.GradientDescentOptimizer(self._lr)
            optimizer = tf.train.AdamOptimizer(self._lr)
            # optimizer = tf.train.GradientDescentOptimizer(0.5)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))  # 将梯度应用于变量

    def add_lstm_cell(self, num_of_cells, keep_prob):
        def lstm_cell(lstm_size, forget_bias=0.0, state_is_tuple=True):
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias, state_is_tuple)
            if keep_prob < 1:  # 在外面包裹一层dropout
                lstm = tf.nn.rnn_cell.DropoutWrapper(
                    lstm, output_keep_prob=self.config.keep_prob)
            return lstm

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(self.config.num_of_hidden_states) for i in range(num_of_cells)],
                                           state_is_tuple=True)  # 多层lstm cell 堆叠起来
        _initial_state = cell.zero_state(self.config.batch_size, data_type())  # 参数初始化,rnn_cell.RNNCell.zero_state
        return cell, _initial_state

    def _fit(self, X, y):
            loss, _ = self.session.run([self.loss, self.train_op],
                                       feed_dict={self.label_: y, self.input: X})
            print(loss)


class MediumConfig(object):
    """Medium config."""
    learning_rate = 0.01
    input_dimension = 3
    batch_size = 100
    weight_scale = 0.1


class LstmConfig(object):
    """Medium config."""
    num_of_stages = 20
    num_of_hidden_states = 400
    num_of_classes = 3
    input_dimension = 14 * 14 * 5
    batch_size = 50
    keep_prob = 0.4
    max_grad_norm = 5
    num_of_cells = 2
    weight_scale = 0.1
    learning_rate = 0.01
    conv1_dimension = 14
    conv2_dimension = 14
    conv3_dimension = 5
    lstm_input_dimension = 50


if __name__ == '__main__':
    from myutils import generate_data,retrieve_return,retrieve_data,dict2ndarray

    data = retrieve_data('医药生物')
    data=dict2ndarray(data)
    thereturn = retrieve_return('医药生物')
    data_train=data[:527,:,:]
    data_test=data[527:,:,:]
    lstm=LSTM(LstmConfig)
    lstm.inference()
    import numpy as np
    for i in range(500):
        X, y = generate_data(data_train, thereturn, 50, 20)
        X=X - X.mean(axis=1).reshape(50, 1, 197, 5)
        X = X[:, :, :-1, :]
        X = X.reshape(50, 20, 196 * 5)
        y = np.argsort(np.argsort(y, axis=2), axis=2)
        y = np.where(y > 130, -2, y)
        y = np.where(y > 65, -1, y)
        y = np.where(y > 0, 0, y)
        y = -y
        y = y[:, :, 0]
        lstm.fit(X, y)

