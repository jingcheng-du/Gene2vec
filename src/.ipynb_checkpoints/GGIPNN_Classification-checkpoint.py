import tensorflow as tf
import numpy as np
import GGIPNN as GGIPNN
import GGIPNN_util as NN_util
import random
import os
import time
import datetime
from sklearn import metrics

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("embedding_file", "../pre_trained_emb/gene2vec_dim_200_iter_9.txt", "embedding file address, matrix txt file")
# Model Hyperparameters
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("embedding_dimension", 200, "embedding dimension")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("use_pre_trained_gene2vec", True, "use_pre_trained_gene2vec") # if False, the embedding layer will be initialized randomly
tf.flags.DEFINE_boolean("train_embedding", False, "train_embedding") # if True, the embedding layer will be trained during the training

FLAGS = tf.app.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

# Data loading data
x_train_raw_f = open("../predictionData/train_text.txt", 'r')
x_train_raw = x_train_raw_f.read().splitlines()
x_train_raw_f.close()
y_train_raw_f = open("../predictionData/train_label.txt", 'r')
y_train_raw = y_train_raw_f.read().splitlines()
y_train_raw_f.close()

x_test_raw_f = open("../predictionData/test_text.txt", 'r')
x_test_raw = x_test_raw_f.read().splitlines()
x_test_raw_f.close()
y_test_raw_f = open("../predictionData/test_label.txt", 'r')
y_test_raw = y_test_raw_f.read().splitlines()
y_test_raw_f.close()

x_valid_raw_f = open("../predictionData/valid_text.txt", 'r')
x_valid_raw = x_valid_raw_f.read().splitlines()
x_valid_raw_f.close()
y_valid_raw_f = open("../predictionData/valid_label.txt", 'r')
y_valid_raw = y_valid_raw_f.read().splitlines()
y_valid_raw_f.close()

all_text_voca = NN_util.myFitDict(x_train_raw+x_valid_raw+x_test_raw,2)
all_text_unshuffled = NN_util.myFit(x_train_raw+x_valid_raw+x_test_raw,2, all_text_voca)

x_train_len = len(x_train_raw)
x_valid_len = len(x_valid_raw)
x_test_len = len(x_test_raw)

x_train_unshuffled= all_text_unshuffled[:x_train_len]
x_valid_unshuffled= all_text_unshuffled[x_train_len:x_train_len+x_valid_len]
x_test_unshuffled= all_text_unshuffled[x_train_len+x_valid_len:]

total = x_train_raw+x_valid_raw+x_test_raw

# Randomly shuffle training data
random_indices = list(range(len(x_train_raw)))
random.shuffle(random_indices)
x_train = x_train_unshuffled[random_indices]
y_onehot = NN_util.oneHot(y_train_raw+y_valid_raw+y_test_raw)

y_train_onehot = y_onehot[:x_train_len]
y_train = y_train_onehot[random_indices]

x_dev = x_valid_unshuffled
y_dev = y_onehot[x_train_len:x_train_len+x_valid_len]

x_test = x_test_unshuffled
y_test = y_onehot[x_train_len+x_valid_len:]

all_predictions = []
all_score = []
all_y2 = []

predicationGS = y_test

print("total training size: " +str(len(y_train)))

print("total test size: " +str(len(y_test)))

print( "training start!")
print("Vocabulary Size: {:d}".format(len(all_text_voca)))
print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
print("GS size " + str(len(predicationGS)))


# Training
# ==================================================
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = GGIPNN.GGIPNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(all_text_voca),
            embedTrain=FLAGS.train_embedding,
            embedding_size = FLAGS.embedding_dimension,
            l2_lambda=FLAGS.l2_reg_lambda
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "../runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.use_pre_trained_gene2vec:
            vocabulary = all_text_voca
            initW = None
            initW = NN_util.load_embedding_vectors(vocabulary,FLAGS.embedding_file,FLAGS.embedding_dimension)
            print("gene embedding file has been loaded\n")
            sess.run(cnn.W.assign(initW))
        else:
            print("embedding loading errors")
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = NN_util.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                # print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                # print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                # print("Saved model checkpoint to {}\n".format(path))

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        # Tensors we want to evaluate
        predictions_score = graph.get_operation_by_name("output/softmax_scores").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        y2 = graph.get_operation_by_name("embedding/embedchar").outputs[0]

        # Generate batches for one epoch
        batches = NN_util.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here

        for x_test_batch in batches:
            batch_score = sess.run(predictions_score, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_score.extend(batch_score)
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions.extend(batch_predictions)
            y2score = sess.run(y2, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_y2.extend(np.transpose(y2score))

predicationGS = np.argmax(predicationGS, axis=1)

predictions_score_human_readable= np.column_stack(np.transpose(all_score))
yscore = predictions_score_human_readable[:,1]

ytrue=predicationGS
print("-------------------")
print("AUC score")
print(metrics.roc_auc_score(np.array(ytrue), np.array(yscore)))