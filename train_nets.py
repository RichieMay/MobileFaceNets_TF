# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

import argparse
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

from losses.face_losses import softmax_loss, arcface_loss
from nets.MobileFaceNet import inference
from utils.common import train
from utils.data_process import parse_function

slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--loss_type', type=str, choices=['softmax', 'arcface'], default=None)
    parser.add_argument('--max_step', type=int, default=800000, help='the max train step')
    parser.add_argument('--num_output', default=85164, help='the train images number')
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--step_schedule', help='Number of epochs for learning rate piecewise.', default=[80000, 100000, 120000, 140000])
    parser.add_argument('--train_batch_size', type=int, default=90, help='batch size to train network')
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str, help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=10, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=500, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=100, help='intervals to show ckpt info')
    parser.add_argument('--pretrained_model', type=str, default='', help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'], help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float, help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms', help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float, help='Loss based on the norm of the activations in the prelogits layer.', default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float, help='Norm to use for prelogits norm loss.', default=1.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args = get_parser()

        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)

        # define placeholder
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = True  # tf.placeholder_with_default(name='phase_train', shape=None, input=tf.constant(False, dtype=tf.bool))

        # prepare train dataset
        # the image is substracted 127.5 and multiplied 1/128.
        # random flip left right
        tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
        dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=args.buffer_size)
        dataset = dataset.batch(args.train_batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        prelogits, net_points = inference(inputs, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)

        # record the network architecture
        hd = open("./arch/txt/MobileFaceNets_Arch.txt", 'w')
        for key in net_points.keys():
            info = '{}:{}\n'.format(key, net_points[key].get_shape().as_list())
            hd.write(info)
        hd.close()

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + 1e-5, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        if args.loss_type == 'softmax':
            inference_loss, logit = softmax_loss(prelogits, labels, args.num_output)
        elif args.loss_type == 'arcface':
            inference_loss, logit = arcface_loss(embeddings, labels, args.num_output, w_init=slim.initializers.xavier_initializer())
        else:
            print('loss_type is invalid, you can use softmax or arcface.')
            exit(0)

        tf.add_to_collection('losses', inference_loss)

        # total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=args.step_schedule, values=[0.001, 0.0005, 0.0003, 0.0001, 0.00001], name='lr_schedule')
        
        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(args.summary_path, sess.graph)
        summaries = []
        # add train info to tensorboard summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summary_op = tf.summary.merge(summaries)

        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')

        # record trainable variable
        hd = open("./arch/txt/trainable_var.txt", "w")
        for var in tf.trainable_variables():
            hd.write(str(var))
            hd.write('\n')
        hd.close()

        # saver to load pretrained model or save model
        # MobileFaceNets_var = [var for var in tf.trainable_variables() if var.name.startswith('MobileFaceNet')]
        # saver_mobilefacenets = tf.train.Saver(MobileFaceNets_var, max_to_keep=args.saver_maxkeep)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        pre_train_step = 0
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            # saver_mobilefacenets.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            ckpt_prefix_name = ckpt.model_checkpoint_path.split('.ckpt')[0]
            split_array = ckpt_prefix_name.split('_')
            pre_train_step = int(split_array[1])

        # output file path
        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)        
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)
        
        print('\nstart training...')
        total_accuracy = {}

        assign_global_step = tf.assign(global_step, pre_train_step, name='assignment_global_step')
        global_step_val = sess.run(assign_global_step)

        while (global_step_val + 1) < args.max_step:
            sess.run(iterator.initializer)
            while (global_step_val + 1) < args.max_step:
                try:
                    images_train, labels_train = sess.run(next_element)
                    feed_dict = {inputs: images_train, labels: labels_train}

                    # print training information
                    global_step_val = global_step_val + 1
                    if global_step_val % args.show_info_interval == 0:
                        start = time.time()

                        _, learning_rate_val, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                            sess.run([train_op, learning_rate, total_loss, inference_loss, regularization_losses, inc_global_step_op, Accuracy_Op],
                                     feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

                        end = time.time()
                        pre_sec = args.train_batch_size / (end - start)

                        print('learning_rate %f, total_step %d, total_loss is %.2f, inference_loss is %.2f, reg_loss is %.2f, train_accuracy is %.6f, '
                              'speed %.3f samples/sec' % (learning_rate_val, global_step_val, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))
                    else:
                        sess.run([train_op, inc_global_step_op], feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

                    # save summary
                    if global_step_val % args.summary_interval == 0:
                        feed_dict = {inputs: images_train, labels: labels_train}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, global_step_val)

                    # save ckpt files
                    if global_step_val % args.ckpt_interval == 0:
                        filename = 'MobileFaceNets_{:d}'.format(global_step_val) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)

                except tf.errors.OutOfRangeError:
                    print("End of total step %d" % global_step_val)
                    break

        summary.close()
        sess.close()
