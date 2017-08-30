import tensorflow as tf
import time
import numpy as np

def data_type():
    return tf.float32

LSTM_NUM=2
WEIGHT_SCARE=0.01

class stock_lstm:
    def __init__(self,num_of_stages,num_of_hidden_states,num_of_classes,input_dimension,batch_size,keep_prob=1,max_grad_norm=5):
        self.num_of_stages=num_of_stages
        self.num_of_hidden_states=num_of_hidden_states
        self.num_of_classes=num_of_classes
        self.input_dimension=input_dimension
        self.batch_size=batch_size
        self.keep_prob=keep_prob
        self.inferenced=False
        self.max_grad_norm=max_grad_norm
        self.graph=tf.Graph()
        self.session=tf.Session()
        self.initialized=False

    def add_place_holder(self):
        """add placeholder and onehot encode"""
        self.input=tf.placeholder(tf.float32,shape=[self.batch_size,self.num_of_stages,self.input_dimension])
        self.label_=tf.placeholder(tf.int32,shape=[self.batch_size,self.num_of_stages])
        self.label=tf.one_hot(self.label_,depth=self.num_of_classes)

    def add_lstm_cell(self,num_of_cells):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_of_hidden_states, forget_bias=0.0, state_is_tuple=True)
        if  self.keep_prob < 1:  # 在外面包裹一层dropout
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_of_cells, state_is_tuple=True)  # 多层lstm cell 堆叠起来
        self._initial_state = cell.zero_state(self.batch_size, data_type())  # 参数初始化,rnn_cell.RNNCell.zero_state
        return cell,self._initial_state

    def inference(self,num_of_cells):
        if self.inferenced:
            return
        self.add_place_holder()
        cell,_initial_state=self.add_lstm_cell(num_of_cells)
        outputs = []
        state = _initial_state
        inputs=self.input

        with tf.variable_scope("RNN"):
            for time_step in range(self.num_of_stages):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # cell_out: [batch, hidden_size]
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)  # output: shape[num_steps][batch,hidden_size]
        output=tf.concat(outputs,axis=1)
        output = tf.reshape(output, [-1, self.num_of_hidden_states])
        softmax_w = tf.get_variable(
            "softmax_w", [self.num_of_hidden_states,self.num_of_classes], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.num_of_classes], dtype=data_type())
        # [batch*numsteps, vocab_size] 从隐藏语义转化成完全表示
        logits = tf.matmul(output, softmax_w) + softmax_b

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],  # output [batch*numsteps, vocab_size]
            [tf.reshape(self.label_, [-1])],  # target, [batch_size, num_steps] 然后展开成一维【列表】
            [tf.ones([self.batch_size * self.num_of_stages], dtype=data_type())])  # weight
        self.loss = tf.reduce_sum(loss) / self.batch_size  # 计算得到平均每批batch的误差
        self._final_state = state
        self.inferenced=True
        self.logits=logits

    def train_epoch(self):
        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                self.add_train_op(2)
                # summary_writer = tf.train.SummaryWriter('/tmp/lstm_logs', session.graph)
                tf.global_variables_initializer().run(session=self.session)
                input = np.random.randn(5, 5, 5)
                label_ = np.random.randint(0, 3, (5, 5))
                for i in range(20000):
                    loss, _ = session.run([self.loss, self.train_op],
                                          feed_dict={self.label_: label_, self.input: input})
                    print(loss)
                logits = session.run(self.logits, feed_dict={self.label_: label_, self.input: input})
                print(np.argmax(logits, axis=1).reshape((5, 5)))
                print(label_)

    def fit(self,X,y):
        session=self.session
        with self.graph.as_default():
            if not self.initialized:
                initializer = tf.random_uniform_initializer(-WEIGHT_SCARE, WEIGHT_SCARE)
                with tf.variable_scope("model", reuse=None, initializer=initializer):
                    self.add_train_op(LSTM_NUM)
                    tf.global_variables_initializer().run(session=session)
                self.initialized=True

            for i in range(20000):
                loss, _ = session.run([self.loss, self.train_op],
                                      feed_dict={self.label_: y, self.input: X})
                print(loss)
            logits = session.run(mylstm.logits, feed_dict={self.label_: label_, self.input: input})
            np.argmax(logits, axis=1).reshape((5, 5))

    def add_train_op(self,num_of_cells):
        self.inference(num_of_cells)

        self._lr = tf.Variable(0.01, trainable=False)
        tvars = tf.trainable_variables()
        # clip_by_global_norm: 梯度衰减，具体算法为t_list[i] * clip_norm / max(global_norm, clip_norm)
        # 这里gradients求导，ys和xs都是张量
        # 返回一个长为len(xs)的张量，其中的每个元素都是\grad{\frac{dy}{dx}}
        # clip_by_global_norm 用于控制梯度膨胀,前两个参数t_list, global_norm, 则
        # t_list[i] * clip_norm / max(global_norm, clip_norm)
        # 其中 global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          self.max_grad_norm)

        # 梯度下降优化，指定学习速率
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # optimizer = tf.train.AdamOptimizer()
        # optimizer = tf.train.GradientDescentOptimizer(0.5)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))  # 将梯度应用于变量

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")  # 用于外部向graph输入新的 lr值
        self._lr_update = tf.assign(self._lr, self._new_lr)  # 使用new_lr来更新lr的值
        return self._train_op

    def assign_lr(self, session, lr_value):
        # 使用 session 来调用 lr_update 操作
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self.input

    @property
    def targets(self):
        return self.label

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self.loss

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    # epoch_size 表示批次总数。也就是说，需要向session喂这么多次数据
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps  # // 表示整数除法
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):
        fetches = [model.cost, model.final_state, eval_op] # 要进行的操作，注意训练时和其他时候eval_op的区别
        feed_dict = {}      # 设定input和target的值
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c   # 这部分有什么用？看不懂
            feed_dict[h] = state[i].h
        cost, state, _ = session.run(fetches, feed_dict) # 运行session,获得cost和state
        costs += cost   # 将 cost 累积
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:  # 也就是每个epoch要输出10个perplexity值
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)



if __name__=='__main__':
    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-0.1,0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mylstm = stock_lstm(5, 5, 5, 5, 5, 0.7)
            mylstm.add_train_op(2)
            # summary_writer = tf.train.SummaryWriter('/tmp/lstm_logs', session.graph)
            tf.initialize_all_variables().run()
            input=np.random.randn(5,5,5)
            label_=np.random.randint(0,3,(5,5))
            for i in range(20000):
                loss,_=session.run([mylstm.loss,mylstm.train_op],feed_dict={mylstm.label_:label_,mylstm.input:input})
                print(loss)
            logits = session.run(mylstm.logits, feed_dict={mylstm.label_: label_, mylstm.input: input})
            np.argmax(logits,axis=1).reshape((5,5))


import tushare as ts
data=ts.get_k_data('600233',ktype='5').head(225).values[:,1:-1].astype(np.float)
data=(data-data.mean(axis=0))/data.std(axis=0)
chanels=np.split(data,5,axis=1)
data=np.concatenate(list(map(lambda x:x.reshape(15,15,1),chanels)),axis=-1)

def get_data(stockId):
    data = ts.get_k_data(stockId, ktype='5').head(225).values[:, 1:-1].astype(np.float)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    chanels = np.split(data, 5, axis=1)
    data = np.concatenate(list(map(lambda x: x.reshape(15, 15, 1), chanels)), axis=-1)
    return data

get_data('600233')


inputss = np.random.randn(5, 5, 5)
label_ = np.random.randint(0, 3, (5, 5))
mylstm = stock_lstm(5, 5, 5, 5, 5, 0.7)
mylstm.fit(input,label_)
