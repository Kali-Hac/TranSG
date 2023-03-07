import numpy as np
import tensorflow as tf
import os, sys
from utils import process_SG as process
from tensorflow.python.layers.core import Dense
from sklearn.preprocessing import label_binarize
import torch
import collections
from sklearn.metrics import average_precision_score
from sklearn import metrics as mr
import gc
import copy
import random

dataset = ''
probe = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

nb_nodes = 20
ft_size = 3  # originial node feature dimension (D)
time_step = 6  # sequence length (f)

# training params
batch_size = 256
nb_epochs = 100000
patience = 100  # patience for early stopping

tf.app.flags.DEFINE_string('dataset', 'KS20', "Dataset: IAS, KS20, BIWI, CASIA-B or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10")  # sequence length (f)
tf.app.flags.DEFINE_string('lr', '0.00035', "learning rate")
tf.app.flags.DEFINE_string('probe', 'probe',
						   "for testing probe")  # "probe" (for KGBD/KS20), "A", "B" (for IAS), "Walking", "Still" (for BIWI)
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('probe_type', '', "probe.gallery")  # probe and gallery setting for CASIA-B
tf.app.flags.DEFINE_string('patience', '120', "epochs for early stopping")
tf.app.flags.DEFINE_string('mode', 'Train', "Training (Train) or Evaluation (Eval)")
tf.app.flags.DEFINE_string('save_flag', '0',
						   "")  # save model metrics (top-1, top-5. top-10, mAP, GPC loss, STPR loss, mACT, mRCL)
tf.app.flags.DEFINE_string('save_model', '0', "")  # save best model
tf.app.flags.DEFINE_string('batch_size', '256', "")
tf.app.flags.DEFINE_string('model_size', '0', "")  # output model size and computational complexity

tf.app.flags.DEFINE_string('H', '128', "")  # embedding size for node representations
tf.app.flags.DEFINE_string('n_heads', '8', "")  # number of Full-Relation (FR) heads
tf.app.flags.DEFINE_string('L_transformer', '2', "")  # number of SGT layers
tf.app.flags.DEFINE_string('seq_lambda', '0.5', "")  # alpha for fusing sequence-level and skeleton-level GPC
tf.app.flags.DEFINE_string('prompt_lambda', '0.5',
						   "")  # beta for fusing structure prompted and trajectory prompted reconstruction
tf.app.flags.DEFINE_string('GPC_lambda', '0.5', "")  # lambda for fusing GPC and STPR
tf.app.flags.DEFINE_string('t_1', '0.07', "")  # global temperatures t1
tf.app.flags.DEFINE_string('t_2', '14', "")  # global temperatures t2
tf.app.flags.DEFINE_string('pos_enc', '1', "")  # positional encoding or not
tf.app.flags.DEFINE_string('enc_k', '10', "")  # first K eigenvectors for positional encoding
tf.app.flags.DEFINE_string('rand_flip', '1', "")  # random flipping strategy
tf.app.flags.DEFINE_string('St_mask_num', '10',
						   "")  # number of random masks (from 1 to (J-1)) in structure prompted reconstruction
tf.app.flags.DEFINE_string('Tr_mask_num', '2',
						   "")  # number of random masks (from 1 to (f-1)) in trajectory prompted reconstruction
tf.app.flags.DEFINE_string('St_prompt_type', '3',
						   "1 (full context), 2 (random (limited num)), 3 (random (pre-defined num))")  # structure masking strategy
tf.app.flags.DEFINE_string('T_prompt', 'l1',
						   "l1, l2, MSE")  # reconstruction loss type for trajectory prompted reconstruction
tf.app.flags.DEFINE_string('S_prompt', 'l1',
						   "l1 or l2 or MSE")  # reconstruction loss type for structure prompted reconstruction
tf.app.flags.DEFINE_string('level', 'J', "J, P, B")  # joint-scale (original), part-scale, or body-scale graphs

FLAGS = tf.app.flags.FLAGS

# check parameters
if FLAGS.dataset not in ['IAS', 'KGBD', 'KS20', 'BIWI', 'CASIA_B']:
	raise Exception('Dataset must be IAS, KGBD, KS20, BIWI or CASIA B.')
if FLAGS.dataset == 'CASIA_B':
	FLAGS.length = '40'
	if FLAGS.length not in ['40', '50', '60']:
		raise Exception('Length number must be 40, 50 or 60')
else:
	if FLAGS.length not in ['4', '6', '8', '10']:
		raise Exception('Length number must be 4, 6, 8 or 10')
if FLAGS.mode not in ['Train', 'Eval']:
	raise Exception('Mode must be Train or Eval.')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset

# optimal paramters
if dataset == 'KGBD':
	FLAGS.lr = '0.00035'
	FLAGS.rand_flip = '0'
	FLAGS.patience = '60'
	# FLAGS.t_2 = '20'
elif dataset == 'CASIA_B':
	FLAGS.lr = '0.00035'
	FLAGS.rand_flip = '0'
	FLAGS.patience = '60'
else:
	FLAGS.lr = '0.00035'

time_step = int(FLAGS.length)
probe = FLAGS.probe
patience = int(FLAGS.patience)
batch_size = int(FLAGS.batch_size)

# not used
global_att = False
nhood = 1
residual = False
nonlinearity = tf.nn.elu

pre_dir = 'ReID_Models/'
# Customize the [directory] to save models with different hyper-parameters
change = ''

if FLAGS.probe_type != '':
	change += '_CME'

change += '_f_' + FLAGS.length + '_layers_' + FLAGS.L_transformer + '_heads_' + FLAGS.n_heads + \
		  '_alpha_' + FLAGS.seq_lambda + '_beta_' + FLAGS.prompt_lambda  + '_lambda_' + FLAGS.GPC_lambda

try:
	os.mkdir(pre_dir)
except:
	pass

if dataset == 'KS20':
	nb_nodes = 25

if dataset == 'CASIA_B':
	nb_nodes = 14

if FLAGS.dataset == 'CASIA_B':
	FLAGS.length = '40'
	FLAGS.Tr_mask_num = '10'

print('----- Model hyperparams -----')
print('f (sequence length): ' + str(time_step))
print('H (embedding size): ' + FLAGS.H)
print('SGT Layers: ' + FLAGS.L_transformer)
print('FR heads: ' + FLAGS.n_heads)
print('alpha: ' + FLAGS.seq_lambda)
print('beta: ' + FLAGS.prompt_lambda)
print('lambda: ' + FLAGS.GPC_lambda)
print('a (structure): ' + FLAGS.St_mask_num)
print('b (trajectory): ' + FLAGS.Tr_mask_num)
print('t1: ' + FLAGS.t_1)
print('t2: ' + FLAGS.t_2)

print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))

print('patience: ' + FLAGS.patience)
print('Mode: ' + FLAGS.mode)

if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	if dataset == 'CASIA_B':
		print('Probe.Gallery: ', FLAGS.probe_type.split('.')[0], FLAGS.probe_type.split('.')[1])
	else:
		print('Probe: ' + FLAGS.probe)

if FLAGS.probe_type == '':
	if FLAGS.level == 'J':
		X_train_J, _, _, _, _, y_train, X_test_J, _, _, _, _, y_test, \
		adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
			process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
								   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
								   enc_k=int(FLAGS.enc_k))
	elif FLAGS.level == 'P':
		_, X_train_P, _, _, _, y_train, _, X_test_P, _, _, _, y_test, \
		_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
			process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
								   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, )
	elif FLAGS.level == 'B':
		_, _, X_train_B, _, _, y_train, _, _, X_test_B, _, _, y_test, \
		_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
			process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
								   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, )
	del _
	gc.collect()

else:
	from utils import process_cme_SG as process

	if FLAGS.level == 'J':
		X_train_J, _, _, _, _, y_train, X_test_J, _, _, _, _, y_test, \
		adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
			process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
								   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
								   PG_type=FLAGS.probe_type.split('.')[0])
	elif FLAGS.level == 'P':
		_, X_train_P, _, _, _, y_train, _, X_test_P, _, _, _, y_test, \
		_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
			process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
								   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
								   PG_type=FLAGS.probe_type.split('.')[0])
	elif FLAGS.level == 'B':
		_, _, X_train_B, _, _, y_train, _, _, X_test_B, _, _, y_test, \
		_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
			process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
								   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
								   PG_type=FLAGS.probe_type.split('.')[0])

	print('## [Probe].[Gallery]', FLAGS.probe_type)
	del _
	gc.collect()

all_ftr_size = int(FLAGS.H)
loaded_graph = tf.Graph()

if FLAGS.level == 'J':
	joint_num = X_train_J.shape[2]
elif FLAGS.level == 'P':
	joint_num = 10
	adj_J = adj_part
	pos_enc_ori = pos_enc_part
	X_test_J = X_test_P
	X_train_J = X_train_P
elif FLAGS.level == 'B':
	joint_num = 5
	adj_J = adj_body
	pos_enc_ori = pos_enc_body
	X_test_J = X_test_B
	X_train_J = X_train_B

cluster_epochs = 15000
display = 20
if FLAGS.level == 'J':
	k = int(FLAGS.enc_k)
elif FLAGS.level == 'P':
	k = 9
elif FLAGS.level == 'B':
	k = 4

if FLAGS.mode == 'Train':
	loaded_graph = tf.Graph()
	with loaded_graph.as_default():
		with tf.name_scope('Input'):
			J_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, joint_num, ft_size))
			L_eig = tf.placeholder(dtype=tf.float32, shape=(joint_num, k))
			train_flag = tf.placeholder(dtype=tf.bool, shape=())
			pseudo_lab_1 = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_1 = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))
			pseudo_lab_2 = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_2 = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))
			seq_mask = tf.placeholder(dtype=tf.float32, shape=(time_step,))
			node_mask = tf.placeholder(dtype=tf.float32, shape=(joint_num,))

			gt_class_ftr = tf.placeholder(dtype=tf.float32, shape=(nb_classes, all_ftr_size))
			gt_lab = tf.placeholder(dtype=tf.int32, shape=(batch_size,))

		with tf.name_scope("TranSG"), tf.variable_scope("TranSG", reuse=tf.AUTO_REUSE):
			inputs = tf.reshape(J_in, [time_step * batch_size * joint_num, 3])
			outputs = inputs
			outputs = tf.layers.dense(outputs, int(FLAGS.H), activation=tf.nn.relu)
			s_rep = outputs
			s_rep = tf.layers.dense(s_rep, int(FLAGS.H), activation=None)
			s_rep = tf.reshape(s_rep, [-1])
			optimizer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.lr))
			seq_ftr = tf.reshape(s_rep, [batch_size, time_step, joint_num, -1])
			pos_enc = tf.tile(tf.reshape(L_eig, [1, 1, joint_num, k]), [batch_size, time_step, 1, 1])
			pos_enc = tf.layers.dense(pos_enc, int(FLAGS.H), activation=None)
			ori_ftr = seq_ftr
			if FLAGS.pos_enc == '1':
				seq_ftr = seq_ftr + pos_enc
			H = int(FLAGS.H)
			W_head = lambda: tf.Variable(tf.random_normal([H, H // int(FLAGS.n_heads)]))

			h = seq_ftr
			for l in range(int(FLAGS.L_transformer)):
				for i in range(int(FLAGS.n_heads)):
					W_Q = tf.Variable(initial_value=W_head)
					W_K = tf.Variable(initial_value=W_head)
					W_V = tf.Variable(initial_value=W_head)
					Q_h = tf.matmul(h, W_Q)
					K_h = tf.matmul(h, W_K)
					K_h = tf.transpose(K_h, perm=[0, 1, 3, 2])
					# numerical stability to clamp [-5, 5]
					att_scores = tf.nn.softmax(
						tf.clip_by_value(tf.matmul(Q_h, K_h) / np.sqrt(H // int(FLAGS.n_heads)), -5, 5))
					# [batch_size, time_step, joint_num, joint_num]
					V_h = tf.matmul(h, W_V)
					att_scores = tf.tile(tf.reshape(att_scores, [batch_size, time_step, joint_num, joint_num, 1]),
										 [1, 1, 1, 1, H // int(FLAGS.n_heads)])
					aggr_features = tf.reduce_sum(
						att_scores * tf.reshape(V_h, [batch_size, time_step, 1, joint_num, H // int(FLAGS.n_heads)]),
						axis=-2)
					if i == 0:
						concat_features = aggr_features
					else:
						concat_features = tf.concat([concat_features, aggr_features], axis=-1)
				h = concat_features
			print('concat_features (Spatial)', h)

			h = tf.layers.dropout(h, rate=0.5, training=train_flag)
			h = tf.layers.dense(h, H, activation=None)
			h_res1 = seq_ftr
			h = h_res1 + h
			h = tf.layers.batch_normalization(h, training=train_flag)
			h_res2 = h
			h = tf.layers.dense(h, H * 2, activation=tf.nn.relu)
			h = tf.layers.dropout(h, rate=0.5, training=train_flag)
			h = tf.layers.dense(h, H, activation=None)
			h = h_res2 + h
			h = tf.layers.batch_normalization(h, training=train_flag)

			gt_pos = tf.reshape(J_in, [batch_size, time_step, joint_num * 3])
			G_recon_loss = 0
			if FLAGS.S_prompt != '0':
				mask_G = tf.boolean_mask(h, tf.reshape(node_mask, [-1]), axis=-2)

				G_h = tf.reduce_mean(mask_G, axis=-2)
				W_r1 = tf.Variable(tf.random_normal([H, H]))
				b_r1 = tf.Variable(tf.zeros(shape=[H, ]))
				h_r1 = tf.matmul(G_h, W_r1) + b_r1
				h_r1 = tf.nn.relu(h_r1)
				W_r2 = tf.Variable(tf.random_normal([H, joint_num * 3]))
				b_r2 = tf.Variable(tf.zeros(shape=[joint_num * 3, ]))
				G_h_r2 = tf.matmul(h_r1, W_r2) + b_r2

				if FLAGS.S_prompt == 'l1':
					G_recon_loss = tf.losses.absolute_difference(G_h_r2, gt_pos) / batch_size
				elif FLAGS.S_prompt == 'l2':
					G_recon_loss = tf.nn.l2_loss(G_h_r2 - gt_pos) / batch_size
				elif FLAGS.S_prompt == 'MSE':
					G_recon_loss = tf.losses.mean_squared_error(G_h_r2, gt_pos) / batch_size

			h_ori = h

			def GPC_ske(t, labels, all_ftr, cluster_ftr):
				W_head = lambda: tf.Variable(tf.random_normal([H, H]))
				head_num = 1
				for i in range(head_num):
					f_1 = tf.Variable(initial_value=W_head)
					f_2 = tf.Variable(initial_value=W_head)
					all_ftr_trans = tf.matmul(all_ftr, f_1)
					cluster_ftr_trans = tf.matmul(cluster_ftr, f_2)
					logits = tf.matmul(all_ftr_trans, tf.transpose(cluster_ftr_trans)) / t
					label_frames = tf.reshape(tf.tile(tf.reshape(labels, [-1, 1]), [1, time_step]), [-1])
					label_frames = tf.reshape(label_frames, [batch_size, time_step])
					loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_frames,
																					logits=logits), axis=-1))
				return loss
			# mean_nodes -> graph representations
			h = tf.reduce_mean(h, axis=-2)
			h2 = h
			C_seq = seq_ftr = h2

			seq_ftr = tf.reduce_mean(seq_ftr, axis=1)
			seq_ftr = tf.reshape(seq_ftr, [batch_size, -1])
			mask_seq_1 = tf.boolean_mask(C_seq, tf.reshape(seq_mask, [-1]), axis=1)
			# # used for trajectory prompting
			mask_seq_1 = tf.reduce_mean(mask_seq_1, axis=1)
			if FLAGS.T_prompt != '0':
				used_f_num = time_step - int(FLAGS.Tr_mask_num)
				T_dim = used_f_num * H
				part_T_enc_1 = tf.boolean_mask(h_ori, tf.reshape(seq_mask, [-1]), axis=1)
				# [batch_size, time_step - mask_num, joint_num, H] -> [batch_size, joint_num, time_step - mask_num, H]
				part_T_enc_1 = tf.transpose(part_T_enc_1, [0, 2, 1, 3])
				part_T_enc_1 = tf.reduce_mean(part_T_enc_1, axis=-2)
				seq_h = tf.reduce_mean(part_T_enc_1, axis=-2)
				W_Tr1 = tf.Variable(tf.random_normal([H, H // 2]))
				b_Tr1 = tf.Variable(tf.zeros(shape=[H // 2, ]))
				h_seq1 = tf.matmul(part_T_enc_1, W_Tr1) + b_Tr1
				h_seq1 = tf.nn.relu(h_seq1)
				W_Tr2 = tf.Variable(tf.random_normal([H // 2, time_step * 3]))
				b_Tr2 = tf.Variable(tf.zeros(shape=[time_step * 3]))
				pred_seq1 = tf.matmul(h_seq1, W_Tr2) + b_Tr2
				T_gt_pos = tf.reshape(gt_pos, [batch_size, time_step, joint_num, 3])
				T_gt_pos = tf.transpose(T_gt_pos, [0, 2, 1, 3])
				T_gt_pos = tf.reshape(T_gt_pos, [batch_size, joint_num, -1])

				if FLAGS.T_prompt == 'l1':
					seq_recon_loss_1 = tf.losses.absolute_difference(pred_seq1, T_gt_pos) / batch_size
				elif FLAGS.T_prompt == 'l2':
					seq_recon_loss_1 = tf.nn.l2_loss(pred_seq1 - T_gt_pos) / batch_size
				elif FLAGS.T_prompt == 'MSE':
					seq_recon_loss_1 = tf.losses.mean_squared_error(pred_seq1, T_gt_pos) / batch_size
				seq_recon_loss = seq_recon_loss_1

			def GPC_seq(t, pseudo_lab, all_ftr, cluster_ftr):
				all_ftr = tf.nn.l2_normalize(all_ftr, axis=-1)
				cluster_ftr = tf.nn.l2_normalize(cluster_ftr, axis=-1)
				output = tf.matmul(all_ftr, tf.transpose(cluster_ftr))
				output /= t
				loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab, logits=output))
				return loss

			if FLAGS.T_prompt != '0':
				recon_loss = G_recon_loss * float(FLAGS.prompt_lambda) + seq_recon_loss * (
							1 - float(FLAGS.prompt_lambda))
			else:
				recon_loss = G_recon_loss

			GPC_seq_loss = GPC_seq(float(FLAGS.t_1), gt_lab, seq_ftr, gt_class_ftr)
			GPC_ske_loss = GPC_ske(float(FLAGS.t_2), gt_lab, C_seq, gt_class_ftr)
			H_loss = (1 - float(FLAGS.seq_lambda)) * GPC_ske_loss + float(FLAGS.seq_lambda) * GPC_seq_loss

			train_op = optimizer.minimize(float(FLAGS.GPC_lambda) * H_loss + (1 - float(FLAGS.GPC_lambda)) * recon_loss)

			if FLAGS.save_flag == '1':
				# Uniform loss
				seq_ftr_norm = tf.nn.l2_normalize(seq_ftr, axis=-1)
				t1 = tf.reshape(seq_ftr_norm, [1, batch_size, -1])
				t2 = tf.reshape(seq_ftr_norm, [batch_size, 1, -1])
				dis_m = tf.norm(t1 - t2, ord='euclidean', axis=-1)
				ones = tf.ones_like(dis_m)
				mask_a = tf.matrix_band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
				mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
				mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
				pairwise_dis = upper_triangular_mat = tf.boolean_mask(dis_m, mask)
				loss_uniform = tf.log(tf.reduce_mean(tf.exp(-2 * tf.square(pairwise_dis))))

		saver = tf.train.Saver()
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with tf.Session(config=config) as sess:
			sess.run(init_op)
			if FLAGS.model_size == '1':
				# compute model size (M) and computational complexity (GFLOPs)
				def stats_graph(graph):
					flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
					params = tf.profiler.profile(graph,
												 options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
					print('FLOPs: {} GFLOPS;    Trainable params: {} M'.format(flops.total_float_ops / 1e9,
																			   params.total_parameters / 1e6))
				stats_graph(loaded_graph)
				exit()

			mask_rand_save = []
			node_mask_save = []

			def train_loader(X_train_J, y_train):
				global mask_rand_save, node_mask_save
				# trajectory masking
				mask_rand_save = []
				# structure masking
				node_mask_save = []
				tr_step = 0
				tr_size = X_train_J.shape[0]
				train_labels_all = []
				train_features_all = []
				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
					rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.Tr_mask_num)),
												   replace=False)
					mask_rand_1 = np.zeros((time_step), dtype=bool)
					mask_rand_1[rand_choice] = True
					rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.Tr_mask_num)),
												   replace=False)
					mask_rand_1 = np.reshape(mask_rand_1, [time_step])
					mask_rand_save.append(mask_rand_1.tolist())

					if FLAGS.St_prompt_type == '1' or FLAGS.St_prompt_type == '2':
						# random node masking with structure prompting
						adj_M = np.reshape(adj_J, [joint_num, joint_num])
						node_flag = [i for i in range(joint_num)]
						rand_list = np.random.choice(joint_num, size=joint_num, replace=False)
						select_nodes = []
						cnt = 0
						while np.mean(node_flag) != -1:
							if node_flag[rand_list[cnt]] != -1:
								select_nodes.append(rand_list[cnt])
								node_flag[rand_list[cnt]] = -1
								for k in range(joint_num):
									if int(adj_M[rand_list[cnt], k]) == 1:
										node_flag[k] = -1
							cnt += 1
						node_mask_rand = np.ones((joint_num), dtype=bool)
						if FLAGS.St_prompt_type == '1':
							node_mask_rand[select_nodes] = False
						elif FLAGS.St_prompt_type == '2':
							rand_list = np.random.choice(joint_num, size=len(select_nodes), replace=False)
							node_mask_rand[rand_list] = False
					elif FLAGS.St_prompt_type == '3':
						node_mask_rand = np.ones((joint_num), dtype=bool)
						rand_list = np.random.choice(joint_num, size=int(FLAGS.St_mask_num), replace=False)
						node_mask_rand[rand_list] = False
					node_mask_save.append(node_mask_rand.tolist())

					[all_features] = sess.run([seq_ftr],
															  feed_dict={
																  J_in: X_input_J,
																  seq_mask: mask_rand_1,
																  L_eig: pos_enc_ori,
																  train_flag: False
															  })
					train_features_all.extend(all_features.tolist())
					train_labels_all.extend(labels.tolist())
					tr_step += 1

				train_features_all = np.array(train_features_all).astype(np.float32)
				train_features_all = torch.from_numpy(train_features_all)
				return train_features_all, train_labels_all

			def gal_loader(X_train_J, y_train):
				tr_step = 0
				tr_size = X_train_J.shape[0]
				gal_logits_all = []
				gal_labels_all = []
				gal_features_all = []
				embed_1_all = []
				embed_2_all = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
					[Seq_features] = sess.run([seq_ftr],
											  feed_dict={
												  J_in: X_input_J,
												  L_eig: pos_enc_ori,
												  train_flag: False
											  })
					gal_features_all.extend(Seq_features.tolist())
					gal_labels_all.extend(labels.tolist())
					tr_step += 1
				return gal_features_all, gal_labels_all, embed_1_all, embed_2_all

			def evaluation():
				vl_step = 0
				vl_size = X_test_J.shape[0]
				pro_labels_all = []
				pro_features_all = []
				while vl_step * batch_size < vl_size:
					if (vl_step + 1) * batch_size > vl_size:
						break
					X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
					[Seq_features] = sess.run([seq_ftr],
											  feed_dict={
												  J_in: X_input_J,
												  L_eig: pos_enc_ori,
												  train_flag: False
											  })
					pro_labels_all.extend(labels.tolist())
					pro_features_all.extend(Seq_features.tolist())
					vl_step += 1
				X = np.array(gal_features_all)
				y = np.array(gal_labels_all)
				t_X = np.array(pro_features_all)
				t_y = np.array(pro_labels_all)
				t_y = np.argmax(t_y, axis=-1)
				y = np.argmax(y, axis=-1)

				def mean_ap(distmat, query_ids=None, gallery_ids=None,
							query_cams=None, gallery_cams=None):
					# distmat = to_numpy(distmat)
					m, n = distmat.shape
					# Fill up default values
					if query_ids is None:
						query_ids = np.arange(m)
					if gallery_ids is None:
						gallery_ids = np.arange(n)
					if query_cams is None:
						query_cams = np.zeros(m).astype(np.int32)
					if gallery_cams is None:
						gallery_cams = np.ones(n).astype(np.int32)
					# Ensure numpy array
					query_ids = np.asarray(query_ids)
					gallery_ids = np.asarray(gallery_ids)
					query_cams = np.asarray(query_cams)
					gallery_cams = np.asarray(gallery_cams)
					# Sort and find correct matches
					indices = np.argsort(distmat, axis=1)
					matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
					# Compute AP for each query
					aps = []
					if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(1, m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
									 (gallery_cams[indices[i]] != query_cams[i]))

							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							# discard nan
							y_score[np.isnan(y_score)] = 0
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					else:
						for i in range(m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
									 (gallery_cams[indices[i]] != query_cams[i]))
							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					if len(aps) == 0:
						raise RuntimeError("No valid query")
					return np.mean(aps)

				def metrics(X, y, t_X, t_y):
					# compute Euclidean distance
					if dataset != 'CASIA_B':
						a, b = torch.from_numpy(t_X), torch.from_numpy(X)
						m, n = a.size(0), b.size(0)
						a = a.view(m, -1)
						b = b.view(n, -1)
						dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
								 torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
						dist_m.addmm_(1, -2, a, b.t())
						# 1e-12
						dist_m = (dist_m.clamp(min=1e-12)).sqrt()
						mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
						_, dist_sort = dist_m.sort(1)
						dist_sort = dist_sort.numpy()
					else:
						X = np.array(X)
						t_X = np.array(t_X)
						dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_m = np.array(dist_m)
						mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
						dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_sort = np.array(dist_sort)

					top_1 = top_5 = top_10 = 0
					probe_num = dist_sort.shape[0]
					if (FLAGS.probe_type == 'nm.nm' or
							FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(probe_num):
							if t_y[i] in y[dist_sort[i, 1:2]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, 1:6]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, 1:11]]:
								top_10 += 1
					else:
						for i in range(probe_num):
							# print(dist_sort[i, :10])
							if t_y[i] in y[dist_sort[i, :1]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, :5]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, :10]]:
								top_10 += 1
					return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

				mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
				del X, y, t_X, t_y, pro_labels_all, pro_features_all
				gc.collect()
				return mAP, top_1, top_5, top_10

			max_acc_1 = 0
			max_acc_2 = 0
			top_5_max = 0
			top_10_max = 0
			cur_patience = 0
			top_1s = []
			top_5s = []
			top_10s = []
			mAPs = []
			h_losses = []
			recon_losses = []
			uni_losses = []

			mACT = []
			mRCL = []

			if dataset == 'KGBD' or dataset == 'KS20':
				if FLAGS.level == 'J':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
				elif FLAGS.level == 'P':
					_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
					_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )
				elif FLAGS.level == 'B':
					_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
					_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )

			elif dataset == 'BIWI':
				if probe == 'Walking':
					if FLAGS.level == 'J':
						X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
						adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
					elif FLAGS.level == 'P':
						_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
						_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )
					elif FLAGS.level == 'B':
						_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
						_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )

				else:
					if FLAGS.level == 'J':
						X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
						adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
					elif FLAGS.level == 'P':
						_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
						_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )
					elif FLAGS.level == 'B':
						_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
						_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )

			elif dataset == 'IAS':
				if probe == 'A':
					if FLAGS.level == 'J':
						X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
						adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
					elif FLAGS.level == 'P':
						_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
						_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )
					elif FLAGS.level == 'B':
						_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
						_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )

				else:
					if FLAGS.level == 'J':
						X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
						adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
					elif FLAGS.level == 'P':
						_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
						_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )
					elif FLAGS.level == 'B':
						_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
						_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
							process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
												   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
												   batch_size=batch_size, )

			elif dataset == 'CASIA_B':
				if FLAGS.level == 'J':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, PG_type=FLAGS.probe_type.split('.')[1])
				elif FLAGS.level == 'P':
					_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
					_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size,
											   PG_type=FLAGS.probe_type.split('.')[1])
				elif FLAGS.level == 'B':
					_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
					_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size,
											   PG_type=FLAGS.probe_type.split('.')[1])
			del _
			gc.collect()
			for epoch in range(cluster_epochs):
				if FLAGS.level == 'P':
					X_gal_J = X_gal_P
				elif FLAGS.level == 'B':
					X_gal_J = X_gal_B
				train_features_all, train_labels_all = train_loader(X_train_J, y_train)
				gal_features_all, gal_labels_all, gal_embed_1_all, gal_embed_2_all = gal_loader(X_gal_J, y_gal)
				ori_train_labels = copy.deepcopy(train_labels_all)
				if FLAGS.save_flag == '1':
					# Compute mean intra-class tightness (mACT) and mean inter-class tightness (mRCL)
					# see "Skeleton Prototype Contrastive Learning with Multi-level Graph Relation Modeling
					# for Unsupervised Person Re-Identification" for details of above metrics
					train_features_all = train_features_all.numpy()
					labels = np.argmax(np.array(train_labels_all), axis=-1)
					label_t = set(labels.tolist())
					y = np.array(labels)
					X = np.array(train_features_all)
					sorted_indices = np.argsort(y, axis=0)
					sort_y = y[sorted_indices]
					sort_X = X[sorted_indices]
					all_class_ftrs = {}
					class_start_indices = {}
					class_end_indices = {}
					pre_label = sort_y[0]
					class_start_indices[pre_label] = 0
					for i, label in enumerate(sort_y):
						if sort_y[i] not in all_class_ftrs.keys():
							all_class_ftrs[sort_y[i]] = [sort_X[i]]
						else:
							all_class_ftrs[sort_y[i]].append(sort_X[i])
						if label != pre_label:
							class_start_indices[label] = class_end_indices[pre_label] = i
							pre_label = label
						if i == len(sort_y) - 1:
							class_end_indices[label] = i
					center_ftrs = []
					for label, class_ftrs in all_class_ftrs.items():
						class_ftrs = np.array(class_ftrs)
						center_ftr = np.mean(class_ftrs, axis=0)
						center_ftrs.append(center_ftr)
					center_ftrs = np.array(center_ftrs)

					a, b = torch.from_numpy(sort_X), torch.from_numpy(center_ftrs)

					a_norm = a / a.norm(dim=1)[:, None]
					b_norm = b / b.norm(dim=1)[:, None]
					dist_m = 1 - torch.mm(a_norm, b_norm.t())
					dist_m = dist_m.numpy()

					prototype_dis_m = np.zeros([nb_classes, nb_classes])
					for i in range(nb_classes):
						prototype_dis_m[i, :] = np.mean(dist_m[class_start_indices[i]:class_end_indices[i], :], axis=0)

					intra_class_dis = np.mean(prototype_dis_m.diagonal())
					sum_distance = np.reshape(np.sum(prototype_dis_m, axis=-1), [nb_classes, ])
					average_distance = np.sum(sum_distance) / (nb_classes * nb_classes)

					a = b = torch.from_numpy(center_ftrs)
					a_norm = a / a.norm(dim=1)[:, None]
					b_norm = b / b.norm(dim=1)[:, None]
					dist_m = 1 - torch.mm(a_norm, b_norm.t())
					dist_m = dist_m.numpy()
					inter_class_dis = np.mean(dist_m)

					mACT.append(average_distance / intra_class_dis)
					mRCL.append(inter_class_dis / average_distance)
					print('mACT: ', average_distance / intra_class_dis, 'mRCL: ', inter_class_dis / average_distance)
					train_features_all = torch.from_numpy(train_features_all)

				mAP, top_1, top_5, top_10 = evaluation()
				cur_patience += 1
				if epoch > 0 and top_1 > max_acc_2:
					max_acc_1 = mAP
					max_acc_2 = top_1
					top_5_max = top_5
					top_10_max = top_10
					try:
						best_cluster_info_1[0] = num_cluster
						best_cluster_info_1[1] = outlier_num
					except:
						pass
					cur_patience = 0
					if FLAGS.mode == 'Train':
						if FLAGS.dataset != 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + 'best.ckpt'
						elif FLAGS.dataset == 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + FLAGS.probe_type + '_best.ckpt'
						print(checkpt_file)
						if FLAGS.save_model == '1':
							saver.save(sess, checkpt_file)
				if epoch > 0:
					if dataset == 'CASIA_B':
						print(
							'[Probe Evaluation] %s - %s | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | mAP: %.4f (%.4f) |' % (
								FLAGS.dataset, FLAGS.probe_type, top_1, max_acc_2, top_5, top_5_max, top_10, top_10_max,
								mAP, max_acc_1))
					else:
						print(
							'[Probe Evaluation] %s - %s | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | mAP: %.4f (%.4f) |' % (
								FLAGS.dataset, FLAGS.probe, top_1, max_acc_2, top_5, top_5_max, top_10, top_10_max, mAP,
								max_acc_1))
					print(
						'%.4f-%.4f-%.4f-%.4f' % (max_acc_2, top_5_max, top_10_max, max_acc_1))
				if cur_patience == patience:
					break

				def generate_cluster_features(labels, features):
					centers = collections.defaultdict(list)
					for i, label in enumerate(labels):
						if label == -1:
							continue
						centers[labels[i]].append(features[i])

					centers = [
						torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
					]
					centers = torch.stack(centers, dim=0)
					return centers

				ori_train_labels = np.array(ori_train_labels)
				ori_train_labels = np.argmax(ori_train_labels, axis=-1)
				y_true = ori_train_labels
				gt_class_features = generate_cluster_features(ori_train_labels, train_features_all)
				X_train_J_new = X_train_J

				tr_step = 0
				tr_size = X_train_J_new.shape[0]

				mask_rand_save = np.array(mask_rand_save)
				node_mask_save = np.array(node_mask_save)
				batch_MPC_loss = []
				batch_MIC_loss = []
				batch_h_loss = []
				batch_recon_loss = []
				batch_uni_loss = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					mask_rand = mask_rand_save[tr_step:(tr_step + 1)]
					mask_rand = np.reshape(mask_rand, [time_step])
					gt_labels = y_true[tr_step * batch_size:(tr_step + 1) * batch_size]

					node_mask_rand = node_mask_save[tr_step:(tr_step + 1)]
					node_mask_rand = np.reshape(node_mask_rand, [joint_num])

					if FLAGS.rand_flip == '1':
						sign_flip = np.random.random(pos_enc_ori.shape[1])
						sign_flip[sign_flip >= 0.5] = 1.0
						sign_flip[sign_flip < 0.5] = -1.0
						pos_enc_ori_rand = pos_enc_ori * sign_flip
						if FLAGS.save_flag == '0':
							_, STPR_loss_, GPC_loss_, Seq_features = sess.run(
								[train_op, recon_loss, H_loss, seq_ftr],
								feed_dict={
									J_in: X_input_J,
									seq_mask: mask_rand,
									node_mask: node_mask_rand,
									L_eig: pos_enc_ori_rand,
									gt_lab: gt_labels,
									gt_class_ftr: gt_class_features,
									train_flag: True
								})
						else:
							_, STPR_loss_, GPC_loss_, uni_loss, Seq_features = sess.run(
								[train_op, recon_loss, H_loss, loss_uniform, seq_ftr],
								feed_dict={
									J_in: X_input_J,
									seq_mask: mask_rand,
									node_mask: node_mask_rand,
									L_eig: pos_enc_ori_rand,
									gt_lab: gt_labels,
									gt_class_ftr: gt_class_features,
									train_flag: True
								})
					elif FLAGS.rand_flip == '0':
						if FLAGS.save_flag == '0':
							_, STPR_loss_, GPC_loss_, Seq_features = sess.run(
								[train_op, recon_loss, H_loss, seq_ftr],
								feed_dict={
									J_in: X_input_J,
									seq_mask: mask_rand,
									node_mask: node_mask_rand,
									L_eig: pos_enc_ori,
									gt_lab: gt_labels,
									gt_class_ftr: gt_class_features,
									train_flag: True
								})
						else:
							_, STPR_loss_, GPC_loss_, uni_loss, Seq_features = sess.run(
								[train_op, recon_loss, H_loss, loss_uniform, seq_ftr],
								feed_dict={
									J_in: X_input_J,
									seq_mask: mask_rand,
									node_mask: node_mask_rand,
									L_eig: pos_enc_ori,
									gt_lab: gt_labels,
									gt_class_ftr: gt_class_features,
									train_flag: True
								})
					batch_h_loss.append(GPC_loss_)
					batch_recon_loss.append(STPR_loss_)
					if FLAGS.save_flag == '1':
						batch_uni_loss.append(uni_loss)

					if tr_step % display == 0:
						print(
							'[%s] Batch num: %d | STPR Loss: %.5f | GPC Loss: %.5f |' %
							(str(epoch), tr_step, STPR_loss_, GPC_loss_))
					tr_step += 1

				h_losses.append(np.mean(batch_h_loss))
				recon_losses.append(np.mean(batch_recon_loss))
				if FLAGS.save_flag == '1':
					uni_losses.append(np.mean(batch_uni_loss))

			if FLAGS.save_flag == '1':
				try:
					os.mkdir(pre_dir + dataset + '/' + probe + change + '/')
				except:
					pass
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'top_1s.npy', top_1s)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'top_5s.npy', top_5s)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'top_10s.npy', top_10s)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mAPs.npy', mAPs)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'hard_loss.npy', h_losses)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'recon_loss.npy', recon_losses)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'uni_loss.npy', uni_losses)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mACT.npy', mACT)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mRCL.npy', mRCL)

			sess.close()

elif FLAGS.mode == 'Eval':
	checkpt_file = pre_dir + FLAGS.dataset + '/' + FLAGS.probe + change + '/best.ckpt'

	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpt_file + '.meta')

		J_in = loaded_graph.get_tensor_by_name("Input/Placeholder:0")
		L_eig = loaded_graph.get_tensor_by_name("Input/Placeholder_1:0")
		train_flag = loaded_graph.get_tensor_by_name("Input/Placeholder_2:0")
		seq_ftr = loaded_graph.get_tensor_by_name("TranSG/TranSG/Reshape_38:0")

		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		loader.restore(sess, checkpt_file)
		saver = tf.train.Saver()
		mask_rand_save = []
		node_mask_save = []

		def gal_loader(X_train_J, y_train):
			tr_step = 0
			tr_size = X_train_J.shape[0]
			gal_logits_all = []
			gal_labels_all = []
			gal_features_all = []
			embed_1_all = []
			embed_2_all = []

			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
				[Seq_features] = sess.run([seq_ftr],
										  feed_dict={
											  J_in: X_input_J,
											  L_eig: pos_enc_ori,
											  train_flag: False
										  })
				gal_features_all.extend(Seq_features.tolist())
				gal_labels_all.extend(labels.tolist())
				tr_step += 1
			return gal_features_all, gal_labels_all, embed_1_all, embed_2_all

		def evaluation():
			vl_step = 0
			vl_size = X_test_J.shape[0]
			pro_labels_all = []
			pro_features_all = []
			while vl_step * batch_size < vl_size:
				if (vl_step + 1) * batch_size > vl_size:
					break
				X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
				[Seq_features] = sess.run([seq_ftr],
										  feed_dict={
											  J_in: X_input_J,
											  L_eig: pos_enc_ori,
											  train_flag: False
										  })
				pro_labels_all.extend(labels.tolist())
				pro_features_all.extend(Seq_features.tolist())
				vl_step += 1
			X = np.array(gal_features_all)
			y = np.array(gal_labels_all)
			t_X = np.array(pro_features_all)
			t_y = np.array(pro_labels_all)
			t_y = np.argmax(t_y, axis=-1)
			y = np.argmax(y, axis=-1)

			def mean_ap(distmat, query_ids=None, gallery_ids=None,
						query_cams=None, gallery_cams=None):
				# distmat = to_numpy(distmat)
				m, n = distmat.shape
				# Fill up default values
				if query_ids is None:
					query_ids = np.arange(m)
				if gallery_ids is None:
					gallery_ids = np.arange(n)
				if query_cams is None:
					query_cams = np.zeros(m).astype(np.int32)
				if gallery_cams is None:
					gallery_cams = np.ones(n).astype(np.int32)
				# Ensure numpy array
				query_ids = np.asarray(query_ids)
				gallery_ids = np.asarray(gallery_ids)
				query_cams = np.asarray(query_cams)
				gallery_cams = np.asarray(gallery_cams)
				# Sort and find correct matches
				indices = np.argsort(distmat, axis=1)
				matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
				# Compute AP for each query
				aps = []
				if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(1, m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
								 (gallery_cams[indices[i]] != query_cams[i]))

						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						# discard nan
						y_score[np.isnan(y_score)] = 0
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				else:
					for i in range(m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
								 (gallery_cams[indices[i]] != query_cams[i]))
						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				if len(aps) == 0:
					raise RuntimeError("No valid query")
				return np.mean(aps)

			def metrics(X, y, t_X, t_y):
				# compute Euclidean distance
				if dataset != 'CASIA_B':
					a, b = torch.from_numpy(t_X), torch.from_numpy(X)
					m, n = a.size(0), b.size(0)
					a = a.view(m, -1)
					b = b.view(n, -1)
					dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
							 torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
					dist_m.addmm_(1, -2, a, b.t())
					# 1e-12
					dist_m = (dist_m.clamp(min=1e-12)).sqrt()
					mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
					_, dist_sort = dist_m.sort(1)
					dist_sort = dist_sort.numpy()
				else:
					X = np.array(X)
					t_X = np.array(t_X)
					dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_m = np.array(dist_m)
					mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
					dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_sort = np.array(dist_sort)

				top_1 = top_5 = top_10 = 0
				probe_num = dist_sort.shape[0]
				if (FLAGS.probe_type == 'nm.nm' or
						FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(probe_num):
						if t_y[i] in y[dist_sort[i, 1:2]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, 1:6]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, 1:11]]:
							top_10 += 1
				else:
					for i in range(probe_num):
						# print(dist_sort[i, :10])
						if t_y[i] in y[dist_sort[i, :1]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, :5]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, :10]]:
							top_10 += 1
				return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

			mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
			del X, y, t_X, t_y, pro_labels_all, pro_features_all
			gc.collect()
			return mAP, top_1, top_5, top_10


		if dataset == 'KGBD' or dataset == 'KS20':
			if FLAGS.level == 'J':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
			elif FLAGS.level == 'P':
				_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
				_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, )
			elif FLAGS.level == 'B':
				_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
				_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, )

		elif dataset == 'BIWI':
			if probe == 'Walking':
				if FLAGS.level == 'J':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
				elif FLAGS.level == 'P':
					_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
					_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )
				elif FLAGS.level == 'B':
					_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
					_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )

			else:
				if FLAGS.level == 'J':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
				elif FLAGS.level == 'P':
					_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
					_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )
				elif FLAGS.level == 'B':
					_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
					_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )

		elif dataset == 'IAS':
			if probe == 'A':
				if FLAGS.level == 'J':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
				elif FLAGS.level == 'P':
					_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
					_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )
				elif FLAGS.level == 'B':
					_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
					_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )

			else:
				if FLAGS.level == 'J':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, enc_k=int(FLAGS.enc_k))
				elif FLAGS.level == 'P':
					_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
					_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )
				elif FLAGS.level == 'B':
					_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
					_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, )

		elif dataset == 'CASIA_B':
			if FLAGS.level == 'J':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, _, pos_enc_ori, _, _, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, PG_type=FLAGS.probe_type.split('.')[1])
			elif FLAGS.level == 'P':
				_, X_train_P, _, _, _, y_train, _, X_gal_P, _, _, _, y_gal, \
				_, _, _, adj_part, _, pos_enc_part, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size,
										   PG_type=FLAGS.probe_type.split('.')[1])
			elif FLAGS.level == 'B':
				_, _, X_train_B, _, _, y_train, _, _, X_gal_B, _, _, y_gal, \
				_, _, _, _, _, _, adj_body, _, pos_enc_body, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size,
										   PG_type=FLAGS.probe_type.split('.')[1])
		del _
		gc.collect()

		mAP_max = top_1_max = top_5_max = top_10_max = 0

		gal_features_all, gal_labels_all, gal_embed_1_all, gal_embed_2_all = gal_loader(X_gal_J, y_gal)
		mAP, top_1, top_5, top_10 = evaluation()
		print(
			'[Evaluation on %s - %s] mAP: %.4f | R1: %.4f - R5: %.4f - R10: %.4f |' %
			(FLAGS.dataset, FLAGS.probe, mAP, top_1, top_5, top_10,))
		sess.close()
		exit()

print('----- Model hyperparams -----')
print('f (sequence length): ' + str(time_step))
print('H (embedding size): ' + FLAGS.H)
print('SGT Layers: ' + FLAGS.L_transformer)
print('FR heads: ' + FLAGS.n_heads)
print('alpha: ' + FLAGS.seq_lambda)
print('beta: ' + FLAGS.prompt_lambda)
print('lambda: ' + FLAGS.GPC_lambda)
print('a (structure): ' + FLAGS.St_mask_num)
print('b (trajectory): ' + FLAGS.Tr_mask_num)
print('t1: ' + FLAGS.t_1)
print('t2: ' + FLAGS.t_2)

print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))

print('patience: ' + FLAGS.patience)
print('Mode: ' + FLAGS.mode)

if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	if dataset == 'CASIA_B':
		print('Probe.Gallery: ', FLAGS.probe_type.split('.')[0], FLAGS.probe_type.split('.')[1])
	else:
		print('Probe: ' + FLAGS.probe)
