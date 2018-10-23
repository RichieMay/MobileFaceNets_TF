# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics

from nets import MobileFaceNet
from utils.data_process import load_data
from verification import evaluate

slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--embedding_size', type=int, help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--test_batch_size', type=int, help='Number of images to process in a batch in the test set.', default=100)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--ckpt_best_path', default='./output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--pretrained_model', type=str, default='./output/ckpt', help='Load a pretrained model before training starts.')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_parser()

    # prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_data(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)

    if not os.path.exists(args.ckpt_best_path):
        os.makedirs(args.ckpt_best_path)

    with tf.Graph().as_default():
        # define placeholder
        inputs = tf.placeholder(name='img_inputs',
                                     shape=[None, args.image_size[0], args.image_size[1], 3],
                                     dtype=tf.float32)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        prelogits, net_points = MobileFaceNet.inference(images=inputs, weight_decay=args.weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping,
                                gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        saver = tf.train.Saver()

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if args.pretrained_model:
            model_checkpoint_path = tf.train.latest_checkpoint(args.pretrained_model)
            saver.restore(sess, model_checkpoint_path)
            print('Restoring pretrained model: %s' % model_checkpoint_path)
        else:
            print('There is no pretrained model')
            exit(0)

        # validate
        print('\nstart testing...')
        for ver_step in range(len(ver_list)):
            start_time = time.time()
            data_sets, issame_list = ver_list[ver_step]
            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
            nrof_batches = data_sets.shape[0] // args.test_batch_size
            for index in range(nrof_batches):  # actual is same multiply 2, test data total
                start_index = index * args.test_batch_size
                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                feed_dict = {inputs: data_sets[start_index:end_index, ...]}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)
            duration = time.time() - start_time

            print("total time %.3f to evaluate %d images of %s" % (duration, data_sets.shape[0], ver_name_list[ver_step]))
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f\n' % eer)

            if ver_name_list == 'lfw' and np.mean(accuracy) > 0.992:
                print('best accuracy is %.5f' % np.mean(accuracy))
                filename = os.path.basename(model_checkpoint_path)
                filename = os.path.join(args.ckpt_best_path, filename)
                saver.save(sess, filename)
