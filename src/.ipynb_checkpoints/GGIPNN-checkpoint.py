import tensorflow as tf

class GGIPNN(object):
    """
    A neural network for text classification.
    Uses an embedding layer, followed by a hidden and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, hidden_dimension=100, embedTrain = False, l2_lambda =0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W", trainable=embedTrain)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x, name = "embedchar")
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # Hidden layer
        with tf.name_scope("layer_1"):
            self.W2 = tf.get_variable(
                "W2",
                shape=[embedding_size*sequence_length, 100],
                initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b2 = tf.Variable(tf.constant(0.1, shape=[hidden_dimension]), name="b2")
            # self.b2 = tf.get_variable(shape=[hidden_dimension], initializer=tf.zeros_initializer(), name="b2")
            # self.y2 = tf.nn.xw_plus_b(tf.reshape(self.embedded_chars,[-1,embedding_size*sequence_length]), self.W2, self.b2)
            self.y2 = tf.nn.relu(tf.nn.xw_plus_b(tf.reshape(self.embedded_chars,[-1,embedding_size*sequence_length]), self.W2, self.b2), name = "ytwo")
        with tf.name_scope("dropout_1"):
            self.h_drop_1 = tf.nn.dropout(self.y2, self.dropout_keep_prob)

        with tf.name_scope("layer_2"):
            self.W3 = tf.get_variable(
                "W3",
                shape=[100, 100],
                initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b3 = tf.Variable(tf.constant(0.1, shape=[100]), name="b3")
            # self.b2 = tf.get_variable(shape=[hidden_dimension], initializer=tf.zeros_initializer(), name="b2")
            # self.y2 = tf.nn.xw_plus_b(tf.reshape(self.embedded_chars,[-1,embedding_size*sequence_length]), self.W2, self.b2)
            self.y3 = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop_1, self.W3, self.b3), name = "ythree")
        with tf.name_scope("dropout_2"):
            self.h_drop_2 = tf.nn.dropout(self.y3, self.dropout_keep_prob)

        with tf.name_scope("layer_3"):
            self.W4 = tf.get_variable(
                "W4",
                shape=[100,10],
                initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b4 = tf.Variable(tf.constant(0.1, shape=[10]), name="b3")
            self.y4 = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop_2, self.W4, self.b4), name = "yfour")
        # Add dropout
        with tf.name_scope("dropout_3"):
            self.h_drop_3 = tf.nn.dropout(self.y4, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W5 = tf.get_variable(
                "W",
                shape=[10, num_classes],
                initializer=tf.contrib.layers.variance_scaling_initializer())
            b5 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop_3, W5, b5, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            self.loss += l2_losses

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # Accuracy