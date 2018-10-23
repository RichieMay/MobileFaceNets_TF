#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author  : RichieMay
# @name    : export_tflite_graph.py
# @time    : 2018/9/20 0020 12:25
import argparse

import tensorflow as tf


def export_tflite_graph():
	input_arrays = ["img_inputs"]
	output_arrays = ["embeddings"]
	converter = tf.contrib.lite.TocoConverter.from_frozen_graph(args.input_file, input_arrays, output_arrays)
	tflite_model = converter.convert()
	open(args.output_file, "wb").write(tflite_model)

def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', type=str, default='../output/frozen_inference_graph.pb',
		help='frozen_inference_graph')
	parser.add_argument('--output_file', type=str,  default='../output/tflite_inference_graph.pb',
		help='tflite_inference_graph')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	export_tflite_graph()

