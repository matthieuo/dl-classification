#    dl-classification
#    Copyright (C) 2017  Matthieu Ospici
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import subprocess
import time
from datetime import datetime

import tensorflow as tf
from sklearn import metrics

import models
import utils
from load_images import create_batch_from_files


def manage_test_process(queue, f, cpopen):
    if queue:
        if cpopen is not None:
            if cpopen.poll() is None:
                return cpopen  # not finish

        print("OK create new process")

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = ""  # test  on CPU

        argument = queue.pop()
        popen = subprocess.Popen(argument, stdout=f, stderr=f, env=my_env)
        return popen
    else:
        return cpopen


def train_model(reg_fact,
                dp_val,
                max_steps,
                data_paths,
                output_path,
                batch_size,
                test_path,
                num_classes,
                ftl):
    with tf.device('/cpu:0'):  # data augmentation on CPU to increase perf
        x, label_batch, _ = create_batch_from_files(
            data_paths, [200, 200], 3, batch_size, ftl, True)

    pred_label = models.foodv_test(
        x,
        num_classes,
        reg_val=reg_fact,
        is_train=True,
        dropout_p=dp_val)

    print(pred_label)

    assert num_classes == ftl.curr_class, "Number of classes found on datasets are not equal to the number of classes given"

    if tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
        reg_losses = tf.add_n(
            tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
    else:
        reg_losses = 0.0

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_label, labels=label_batch, name='cross_entropy')

    # global loss with the cross entropy and the L2 reg
    loss = reg_losses + cross_entropy

    correct_prediction = tf.equal(
        tf.cast(
            tf.argmax(
                pred_label,
                1),
            tf.int32),
        label_batch)
    accuracy_class = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    prediction_label = tf.argmax(pred_label, 1)

    tf.summary.scalar("acc_class", accuracy_class)

    train_operations = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()  # for tensorboard

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(output_path, sess.graph)

        test_process_queue = []
        curr_popen = None

        with open(output_path + "/eval_model_output.log", 'a') as log_file:
            for step in range(max_steps):

                curr_popen = manage_test_process(
                    test_process_queue, log_file, curr_popen)

                start_time = time.time()

                _ = sess.run(train_operations)

                duration = time.time() - start_time

                if step % 100 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.now(), step, -1,
                                         examples_per_sec, sec_per_batch))

                    print("wrote sumary op")

                    sum_op, ac, lb, pb = sess.run(
                        [summary_op, accuracy_class, label_batch, prediction_label])

                    print("ac ", ac)
                    print(lb)
                    print(pb)

                    print(metrics.classification_report(lb, pb))

                    summary_writer.add_summary(sum_op, step)

                if step % 10000 == 0:  # and  step > 0: #save a checkpoint and launch test every 1000 iterations
                    print("sav checkpoint")
                    checkpoint_path = os.path.join(output_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print("done")

                    if test_path:
                        # now launch test on different test sets !
                        print("Enqueue evalution on test set")

                        py_com = ['python3']

                        tests = [('TEST', test_path)]

                        for idt, path in tests:
                            test_args = py_com + ["./eval.py",
                                                  "-paths",
                                                  path,
                                                  "-lp",
                                                  output_path,
                                                  "-ti",
                                                  idt,
                                                  '-nc',
                                                  str(num_classes)]

                            test_process_queue.insert(0, test_args)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    group_genera = parser.add_argument_group('General options')

    group_genera.add_argument(
        "-paths",
        "--data-paths",
        help="Path to the data to load, can be comma separated",
        required=True)

    group_train = parser.add_argument_group('Training control')

    group_train.add_argument(
        "-s",
        "--append-string",
        help="String to append to the log directory",
        required=True)
    group_train.add_argument(
        "-dp",
        "--dropout",
        help="Dropout value, 1 = no dropout",
        type=float,
        required=True)
    group_train.add_argument(
        "-reg",
        "--reg-fact",
        help="L2 regularization factor",
        type=float,
        required=True)
    group_train.add_argument(
        "-bp",
        "--base_path",
        help="Base path for log",
        required=True)
    group_train.add_argument(
        "-nc",
        "--num-classes",
        help="Number of classes on the training set",
        type=int,
        required=True)
    group_train.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size",
        type=int,
        required=True)
    group_train.add_argument(
        "-tp",
        "--test_path",
        help="Path to the test set",
        required=False)
    args = parser.parse_args()

    data_paths = args.data_paths.split(';')
    num_classes = args.num_classes
    reg_fact = args.reg_fact

    if (args.dropout == 1):
        dp_str = "No"
        dropout_f = False
    else:
        dp_str = str(args.dropout)
        dropout_f = True

    output_dir = "_reg_" + str(reg_fact) + "_cla_" + str(num_classes) + "_dp_" + dp_str + "_bs_" + str(args.batch_size) + "_str_" + args.append_string

    output_path = os.path.join(args.base_path, output_dir)

    ftl = utils.file_to_label_binary()

    print("Configuration resume")

    print("++++ data paths   : ", data_paths)
    print("---- output path : ", output_path)

    print("reg fact = ", reg_fact)
    print("dropout = ", dp_str)
    print("output = ", output_path)
    print("Batch size = ", args.batch_size)
    print("Test path = ", args.test_path)
    print("Number of classes = ", num_classes)

    train_model(reg_fact,
                args.dropout,
                30000,
                data_paths,
                output_path,
                args.batch_size,
                args.test_path,
                num_classes,
                ftl)
