import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import copy
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

"""
 Generate training data for each dataset.
"""

def gen_train_data(dataset, split, time_step, nb_nodes, nhood, global_att, batch_size, PG_type, reverse='0'):
	def get_data(dimension, fr):
		if reverse == '1':
			used_data = 'target'
		else:
			used_data = 'source'
		input_data = np.load(
			'Datasets/' + frames_ps + dataset + '_train_npy_data/' + used_data + '_' + dimension + '_' + dataset + '_' + str(
				fr) + '.npy')
		input_data = input_data.reshape([-1, time_step, nb_nodes])
		spine_pos = input_data[:, :, 0]
		spine_pos = np.expand_dims(spine_pos, -1)
		input_data = input_data - spine_pos
		if dataset == 'IAS':
			t_input_data = np.load(
				'Datasets/' + frames_ps + dataset + '_test_npy_data/t_' + used_data + '_' + dimension + '_IAS-' + split + '_' + str(
					fr) + '.npy')
		else:
			t_input_data = np.load(
				'Datasets/' + frames_ps + dataset + '_test_npy_data/' + PG_type + '/t_' + used_data + '_' + dimension + '_' + dataset + '_' + str(
					fr) + '.npy')
		t_input_data = t_input_data.reshape([-1, time_step, nb_nodes])
		# Normalize
		t_spine_pos = t_input_data[:, :, 0]
		t_spine_pos = np.expand_dims(t_spine_pos, -1)
		t_input_data = t_input_data - t_spine_pos

		return input_data, t_input_data

	def unnormalized_laplacian(adj_matrix):
		R = np.sum(adj_matrix, axis=1)
		degreeMatrix = np.diag(R)
		return degreeMatrix - adj_matrix

	def normalized_laplacian(adj_matrix):
		R = np.sum(adj_matrix, axis=1)
		R_sqrt = 1 / np.sqrt(R)
		D_sqrt = np.diag(R_sqrt)
		I = np.eye(adj_matrix.shape[0])
		return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

	# For CASIA B CME
	frames_ps = dataset + '_match/' + str(time_step) + '/'
	input_data_x, t_input_data_x = get_data('x', fr=time_step)
	input_data_y, t_input_data_y = get_data('y', fr=time_step)
	if dataset == 'OUMVLP':
		input_data_z, t_input_data_z = np.zeros_like(input_data_x), np.zeros_like(t_input_data_x)
	else:
		input_data_z, t_input_data_z = get_data('z', fr=time_step)

	X_train = np.concatenate([input_data_x, input_data_y, input_data_z], axis=-1)
	X_test = np.concatenate([t_input_data_x, t_input_data_y, t_input_data_z], axis=-1)

	ids = np.load(
		'Datasets/' + frames_ps + dataset + '_train_npy_data/ids_' + dataset + '_' + str(time_step) + '.npy')
	ids = ids.item()
	if dataset == 'IAS':
		t_ids = np.load(
			'Datasets/' + frames_ps + 'IAS_test_npy_data/ids_IAS-' + split + '_' + str(time_step) + '.npy')
	else:
		t_ids = np.load(
			'Datasets/' + frames_ps + dataset + '_test_npy_data/' + PG_type + '/ids_' + dataset + '_' + str(time_step) + '.npy')
	t_ids = t_ids.item()

	y_train = np.load(
		'Datasets/' + frames_ps + dataset + '_train_npy_data/frame_id_' + dataset + '_' + str(time_step) + '.npy')
	if dataset == 'IAS':
		y_test = np.load(
			'Datasets/' + frames_ps + 'IAS_test_npy_data/frame_id_IAS-' + split + '_' + str(time_step) + '.npy')
	else:
		y_test = np.load(
			'Datasets/' + frames_ps + dataset + '_test_npy_data/' + PG_type + '/frame_id_' + dataset + '_' + str(time_step) + '.npy')

	if dataset != 'OUMVLP':
		X_train, y_train = class_samp_gen(X_train, y_train, ids, batch_size)

	# randomly shuffle
	rand_p = np.random.permutation(X_train.shape[0])
	X_train = X_train[rand_p]
	y_train = y_train[rand_p]
	# print(X_train.shape, y_train.shape)

	ids_keys = sorted(list(ids.keys()))
	classes = [i for i in ids_keys]
	y_train = label_binarize(y_train, classes=classes)
	t_ids_keys = sorted(list(t_ids.keys()))
	classes = [i for i in t_ids_keys]
	y_test = label_binarize(y_test, classes=classes)

	X_train_J = X_train.reshape([-1, time_step, 3, nb_nodes])
	X_train_J = np.transpose(X_train_J, [0, 1, 3, 2])
	X_train_P = reduce2part(X_train_J, nb_nodes)
	X_train_B = reduce2body(X_train_J, nb_nodes)
	X_train_H_B = reduce2h_body(X_train_J, nb_nodes)
	if dataset == 'KS20':
		X_train_In = interpolation(X_train_J, nb_nodes)

	X_test_J = X_test.reshape([-1, time_step, 3, nb_nodes])
	X_test_J = np.transpose(X_test_J, [0, 1, 3, 2])
	X_test_P = reduce2part(X_test_J, nb_nodes)
	X_test_B = reduce2body(X_test_J, nb_nodes)
	X_test_H_B = reduce2h_body(X_test_J, nb_nodes)
	if dataset == 'KS20':
		X_test_In = interpolation(X_test_J, nb_nodes)

	def generate_denser_adj(adj):
		adj_temp = copy.deepcopy(adj).tolist()
		node_num = len(adj_temp)
		new_adj = np.zeros([node_num * 2 - 1, node_num * 2 - 1])
		cnt = node_num
		for i in range(node_num):
			for j in range(node_num):
				if adj_temp[i][j] == 1:
					new_adj[i, cnt] = new_adj[cnt, i] = new_adj[j, cnt] = new_adj[cnt, j] = 1
					adj_temp[i][j] = adj_temp[j][i] = 0
					# print(i, j, cnt)
					cnt += 1
		for i in range(node_num):
			for j in range(node_num):
				if adj_temp[i][j] == 1:
					assert new_adj[i, j] == new_adj[j, i] == 0
		if global_att:
			new_adj = np.ones([node_num*2-1, node_num*2-1])
		# print(cnt)
		# print(new_adj)
		return new_adj
	import scipy.sparse
	if dataset == 'KS20':
		# Joint-Level adjacent matrix
		j_pair_1 = np.array([3, 2, 20, 8, 8, 9, 10, 9, 11, 10, 4, 20, 4, 5, 5, 6, 6, 7, 1, 20, 1, 0, 16, 0,
		                12, 0, 16, 17, 12, 13, 17, 18, 19, 18, 13, 14, 14, 15, 2, 20, 11, 23, 10, 24, 7, 21, 6, 22])
		j_pair_2 = np.array([2, 3, 8, 20, 9, 8, 9, 10, 10, 11, 20, 4, 5, 4, 6, 5, 7, 6, 20, 1, 0, 1, 0, 16,
		                0, 12, 17, 16, 13, 12, 18, 17, 18, 19, 14, 13, 15, 14, 20, 2, 23, 11, 24, 10, 21, 7, 22, 6])
		con_matrix = np.ones([48])
		adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
		if global_att:
			adj_joint = np.ones([25, 25])
		# hyper-Level (interpolation) adjacent matrix, NOT used, replaced by generate_denser_adj
		# i_pair_1 = np.array(
		# 	[3, 31, 31, 2, 2, 32, 32, 20, 20, 42, 42, 8, 8, 37, 37, 9, 9, 44, 44, 10, 10, 38, 38, 11, 11,
		# 	 39, 39, 23, 10, 40, 40, 24, 20, 41, 41, 4, 4, 33, 33, 5, 5, 43, 43, 6, 6, 34, 34, 7, 7, 35,
		# 	 35, 21, 6, 36, 36, 22,
		# 	 20, 45, 45, 1, 1, 30, 30, 0, 0, 46, 46, 12, 12, 26, 26, 13, 13, 48, 48, 14, 14, 27, 27, 15,
		# 	 0, 47, 47, 16, 16, 28, 28, 17, 17, 49, 49, 18, 18, 29, 29, 19])
		# # miss 25
		# i_pair_1 = i_pair_1.tolist()
		# for i in range(len(i_pair_1)):
		# 	if i_pair_1[i] > 24:
		# 		i_pair_1[i] -= 1
		# i_pair_1 = np.array(i_pair_1)
		# i_pair_2 = np.array(
		# 	[31, 3, 2, 31, 32, 2, 20, 32, 42, 20, 8, 42, 37, 8, 9, 37, 44, 9, 10, 44, 38, 10, 11, 38, 39,
		# 	 11, 23, 39, 40, 10, 24, 40, 41, 20, 4, 41, 33, 4, 5, 33, 43, 5, 6, 43, 34, 6, 7, 34, 35, 7,
		# 	 21, 35, 36, 6, 22, 36,
		# 	 45, 20, 1, 45, 30, 1, 0, 30, 46, 0, 12, 46, 26, 12, 13, 26, 48, 13, 14, 48, 27, 14, 15, 27,
		# 	 47, 0, 16, 47, 28, 16, 17, 28, 49, 17, 18, 49, 29, 18, 19, 29])
		# i_pair_2 = i_pair_2.tolist()
		# for i in range(len(i_pair_2)):
		# 	if i_pair_2[i] > 24:
		# 		i_pair_2[i] -= 1
		# i_pair_2 = np.array(i_pair_2)
		# # print(i_pair_1.shape, i_pair_2.shape)
		# con_matrix = np.ones([96])
		# adj_interp = scipy.sparse.coo_matrix((con_matrix, (i_pair_1, i_pair_2)),
		#                                     shape=(49, 49)).toarray()
		adj_interp = generate_denser_adj(adj_joint)


	elif dataset == 'CASIA_B':
		# Joint-Level adjacent matrix
		j_pair_1 = np.array([0, 1, 1, 2, 2, 3, 3, 4, 1, 5, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13])
		j_pair_2 = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 1, 6, 5, 7, 6, 8, 1, 9, 8, 10, 9, 11, 1, 12, 11, 13, 12])
		con_matrix = np.ones([26])
		adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
		if global_att:
			adj_joint = np.ones([14, 14])
		# adj_interp = generate_denser_adj(adj_joint)
	elif dataset == 'OUMVLP':
		# Joint-Level adjacent matrix
		j_pair_1 = np.array([0, 1, 1, 2, 2, 3, 3, 4, 1, 5, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 14,  0, 15,  0,
		                     15, 17, 16, 14])
		j_pair_2 = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 1, 6, 5, 7, 6, 8, 1, 9, 8, 10, 9, 11, 1, 12, 11, 13, 12,  0, 14,  0, 15,
		                     17, 15, 14, 16])
		con_matrix = np.ones([34])
		adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
		if global_att:
			adj_joint = np.ones([18, 18])
		adj_interp = generate_denser_adj(adj_joint)
	else:
		# Joint-Level adjacent matrix
		j_pair_1 = np.array([3, 2, 2, 8, 8, 9, 10, 9, 11, 10, 4, 2, 4, 5, 5, 6, 6, 7, 1, 2, 1, 0, 16, 0,
		                12, 0, 16, 17, 12, 13, 17, 18, 19, 18, 13, 14, 14, 15])
		j_pair_2 = np.array([2, 3, 8, 2, 9, 8, 9, 10, 10, 11, 2, 4, 5, 4, 6, 5, 7, 6, 2, 1, 0, 1, 0, 16,
		                0, 12, 17, 16, 13, 12, 18, 17, 18, 19, 14, 13, 15, 14])
		con_matrix = np.ones([38])
		adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
		if global_att:
			adj_joint = np.ones([20, 20])
		adj_interp = generate_denser_adj(adj_joint)

	# compute Laplacian matrix, use the k smallest non-trivial eigen-vectors
	k = 10
	L = normalized_laplacian(adj_joint)
	# only real
	EigVal, EigVec = np.linalg.eig(L)
	idx = EigVal.argsort()
	EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
	# print(EigVal.shape, EigVec.shape)
	# exit()
	pos_enc_ori = EigVec[:, 1:k + 1]

	# Part-Level adjacent matrix
	p_pair_1 = np.array([5, 6, 5, 8, 6, 7, 8, 9, 5, 4, 4, 2, 4, 0, 2, 3, 1, 0])
	p_pair_2 = np.array([6, 5, 8, 5, 7, 6, 9, 8, 4, 5, 2, 4, 0, 4, 3, 2, 0, 1])
	con_matrix = np.ones([18])
	adj_part = scipy.sparse.coo_matrix((con_matrix, (p_pair_1, p_pair_2)), shape=(10, 10)).toarray()

	# Body-Level adjacent matrix
	b_pair_1 = np.array([2, 3, 2, 4, 2, 1, 2, 0])
	b_pair_2 = np.array([3, 2, 4, 2, 1, 2, 0, 2])
	con_matrix = np.ones([8])
	adj_body = scipy.sparse.coo_matrix((con_matrix, (b_pair_1, b_pair_2)), shape=(5, 5)).toarray()

	# Hyper-Body-Level adjacent matrix
	# h_b_pair_1 = np.array([0, 1, 2, 1])
	# h_b_pair_2 = np.array([1, 0, 1, 2])
	# con_matrix = np.ones([4])
	# adj_hyper_body = scipy.sparse.coo_matrix((con_matrix, (h_b_pair_1, h_b_pair_2)), shape=(3, 3)).toarray()

	# compute Laplacian matrix for part-level and body-level, use the k (1/2 node number) smallest non-trivial eigen-vectors
	# part-level
	k = 9
	L = normalized_laplacian(adj_part)
	# only real
	EigVal, EigVec = np.linalg.eig(L)
	idx = EigVal.argsort()
	EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
	pos_enc_part = EigVec[:, 1:k + 1]

	# body-level
	k = 4
	L = normalized_laplacian(adj_body)
	# only real
	EigVal, EigVec = np.linalg.eig(L)
	idx = EigVal.argsort()
	EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
	pos_enc_body = EigVec[:, 1:k + 1]

	if global_att:
		adj_part = np.ones([10, 10])
		adj_body = np.ones([5, 5])

	# if dataset != 'KS20':
	# X_train_In = generate_denser_graph_data(X_train_J, adj_joint, nb_nodes)
	# X_test_In = generate_denser_graph_data(X_test_J, adj_joint, nb_nodes)

	if dataset == 'IAS':
		nb_classes = 11
	elif dataset == 'KGBD':
		nb_classes = 164
	elif dataset == 'BIWI':
		nb_classes = 28
	elif dataset == 'KS20':
		nb_classes = 20
	elif dataset == 'CASIA_B':
		nb_classes = 62
	elif dataset == 'OUMVLP':
		nb_classes = 10295

	adj_joint = adj_joint[np.newaxis]
	biases_joint = adj_to_bias(adj_joint, [nb_nodes], nhood=nhood)

	# adj_part = adj_part[np.newaxis]
	# biases_part = adj_to_bias(adj_part, [10], nhood=1)
	#
	# adj_body = adj_body[np.newaxis]
	# biases_body = adj_to_bias(adj_body, [5], nhood=1)

	# adj_interp = adj_interp[np.newaxis]
	# biases_interp = adj_to_bias(adj_interp, [nb_nodes*2-1], nhood=1)

	# adj_hyper_body = adj_hyper_body[np.newaxis]
	# biases_hyper_body = adj_to_bias(adj_hyper_body, [3], nhood=1)

	# return X_train_J, X_train_P, X_train_B, X_train_H_B, X_train_In, y_train, X_test_J, X_test_P, X_test_B, X_test_H_B, X_test_In, y_test, \
	# 	       adj_joint, biases_joint, adj_part, biases_part, adj_body, biases_body, adj_hyper_body, biases_hyper_body, adj_interp, biases_interp, nb_classes
	# return X_train_J, X_train_P, X_train_B, 0, 0, y_train, X_test_J, X_test_P, X_test_B, 0, 0, y_test, \
	# 	      adj_joint, biases_joint, pos_enc_ori, 0, 0, 0, 0, 0, 0, 0, 0, nb_classes

	# return X_train_J, 0, 0, 0, 0, y_train, X_test_J, 0, 0, 0, 0, y_test, \
	# 	   adj_joint, biases_joint, pos_enc_ori, 0, 0, 0, 0, 0, 0, 0, 0, nb_classes
	return X_train_J, X_train_P, X_train_B, 0, 0, y_train, X_test_J, X_test_P, X_test_B, 0, 0, y_test, \
						   adj_joint, 0, pos_enc_ori, adj_part, 0, pos_enc_part, adj_body, 0, pos_enc_body, 0, 0, 0, 0, nb_classes

"""
 Generate part-level  skeleton graphs.
"""

def reduce2part(X, joint_num=20):
	if joint_num == 25:
		left_leg_up = [12, 13]
		left_leg_down = [14, 15]
		right_leg_up = [16, 17]
		right_leg_down = [18, 19]
		torso = [0, 1]
		head = [2, 3, 20]
		left_arm_up = [4, 5]
		left_arm_down = [6, 7, 21, 22]
		right_arm_up = [8, 9]
		right_arm_down = [10, 11, 23, 24]
	elif joint_num == 20:
		left_leg_up = [12, 13]
		left_leg_down = [14, 15]
		right_leg_up = [16, 17]
		right_leg_down = [18, 19]
		torso = [0, 1]
		head = [2, 3]
		left_arm_up = [4, 5]
		left_arm_down = [6, 7]
		right_arm_up = [8, 9]
		right_arm_down = [10, 11]
	elif joint_num == 14:
		left_leg_up = [11]
		left_leg_down = [12, 13]
		right_leg_up = [8]
		right_leg_down = [9, 10]
		torso = [1]
		head = [0]
		left_arm_up = [5]
		left_arm_down = [6, 7]
		right_arm_up = [2]
		right_arm_down = [3, 4]
	elif joint_num == 18:
		left_leg_up = [11]
		left_leg_down = [12, 13]
		right_leg_up = [8]
		right_leg_down = [9, 10]
		torso = [1]
		head = [0, 14, 15, 16, 17]
		left_arm_up = [5]
		left_arm_down = [6, 7]
		right_arm_up = [2]
		right_arm_down = [3, 4]

	x_torso = np.mean(X[:, :, torso, :], axis=2)  # [N * T, V=1]
	x_leftlegup = np.mean(X[:, :, left_leg_up, :], axis=2)
	x_leftlegdown = np.mean(X[:, :, left_leg_down, :], axis=2)
	x_rightlegup = np.mean(X[:, :, right_leg_up, :], axis=2)
	x_rightlegdown = np.mean(X[:, :, right_leg_down, :], axis=2)
	x_head = np.mean(X[:, :, head, :], axis=2)
	x_leftarmup = np.mean(X[:, :, left_arm_up, :], axis=2)
	x_leftarmdown = np.mean(X[:, :, left_arm_down, :], axis=2)
	x_rightarmup = np.mean(X[:, :, right_arm_up, :], axis=2)
	x_rightarmdown = np.mean(X[:, :, right_arm_down, :], axis=2)
	X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head, x_leftarmup,
	                         x_leftarmdown, x_rightarmup, x_rightarmdown), axis=-1) \
		.reshape([X.shape[0], X.shape[1], 10, 3])
	return X_part

"""
 Generate body-level  skeleton graphs.
"""

def reduce2body(X, joint_num=20):
	if joint_num == 25:
		left_leg = [12, 13, 14, 15]
		right_leg = [16, 17, 18, 19]
		torso = [0, 1, 2, 3, 20]
		left_arm = [4, 5, 6, 7, 21, 22]
		right_arm = [8, 9, 10, 11, 23, 24]
	elif joint_num == 20:
		left_leg = [12, 13, 14, 15]
		right_leg = [16, 17, 18, 19]
		torso = [0, 1, 2, 3]
		left_arm = [4, 5, 6, 7]
		right_arm = [8, 9, 10, 11]
	elif joint_num == 14:
		left_leg = [11, 12, 13]
		right_leg = [8, 9, 10]
		torso = [0, 1]
		left_arm = [5, 6, 7]
		right_arm = [2, 3, 4]
	elif joint_num == 18:
		left_leg = [11, 12, 13]
		right_leg = [8, 9, 10]
		torso = [0, 1, 14, 15, 16, 17]
		left_arm = [5, 6, 7]
		right_arm = [2, 3, 4]

	x_torso = np.mean(X[:, :, torso, :], axis=2)  # [N * T, V=1]
	x_leftleg = np.mean(X[:, :, left_leg, :], axis=2)
	x_rightleg = np.mean(X[:, :, right_leg, :], axis=2)
	x_leftarm = np.mean(X[:, :, left_arm, :], axis=2)
	x_rightarm = np.mean(X[:, :, right_arm, :], axis=2)
	X_body = np.concatenate((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), axis=-1)\
		.reshape([X.shape[0], X.shape[1], 5, 3])
	return X_body

"""
 Generate hyper-body-level  skeleton graphs.
"""

def reduce2h_body(X, joint_num=20):
	if joint_num == 25:
		# left_leg = [12, 13, 14, 15]
		# right_leg = [16, 17, 18, 19]
		# torso = [0, 1, 2, 3, 20]
		# left_arm = [4, 5, 6, 7, 21, 22]
		# right_arm = [8, 9, 10, 11, 23, 24]
		upper = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24]
		middle = [0]
		lower = [12, 13, 14, 15, 16, 17, 18, 19]
	elif joint_num == 20:
		# left_leg = [12, 13, 14, 15]
		# right_leg = [16, 17, 18, 19]
		# torso = [0, 1, 2, 3]
		# left_arm = [4, 5, 6, 7]
		# right_arm = [8, 9, 10, 11]
		upper = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
		middle = [0]
		lower = [12, 13, 14, 15, 16, 17, 18, 19]
	elif joint_num == 14:
		# left_leg = [11, 12, 13]
		# right_leg = [8, 9, 10]
		# torso = [0, 1]
		# left_arm = [5, 6, 7]
		# right_arm = [2, 3, 4]
		upper = [0, 1, 2, 3, 4, 5, 6, 7]
		middle = [8, 11]
		lower = [9, 10, 12, 13]
	elif joint_num == 18:
		# left_leg = [11, 12, 13]
		# right_leg = [8, 9, 10]
		# torso = [0, 1]
		# left_arm = [5, 6, 7]
		# right_arm = [2, 3, 4]
		upper = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]
		middle = [8, 11]
		lower = [9, 10, 12, 13]

	x_upper = np.mean(X[:, :, upper, :], axis=2)  # [N * T, V=1]
	x_middle = np.mean(X[:, :, middle, :], axis=2)
	x_lower = np.mean(X[:, :, lower, :], axis=2)
	X_body = np.concatenate((x_upper, x_middle, x_lower), axis=-1)\
		.reshape([X.shape[0], X.shape[1], 3, 3])
	return X_body

"""
 Linear interpolation
"""

def interpolation(X, joint_num=20):
	if joint_num == 25:
		left_leg_up = [12, 13]
		left_leg_down = [14, 15]
		right_leg_up = [16, 17]
		right_leg_down = [18, 19]
		torso = [0, 1]
		head_1 = [2, 3]
		head_2 = [2, 20]
		left_arm_up = [4, 5]
		left_arm_down_1 = [6, 7]
		left_arm_down_2 = [7, 21]
		left_arm_down_3 = [6, 22]
		right_arm_up = [8, 9]
		right_arm_down_1 = [10, 11]
		right_arm_down_2 = [11, 23]
		right_arm_down_3 = [10, 24]
		shoulder_1 = [4, 20]
		shoulder_2 = [8, 20]
		elbow_1 = [5, 6]
		elbow_2 = [9, 10]
		spine_mm = [20, 1]
		hip_1 = [0, 12]
		hip_2 = [0, 16]
		knee_1 = [13, 14]
		knee_2 = [17, 18]
		x_torso = np.mean(X[:, :, torso, :], axis=2)  # [N * T, V=1]
		x_leftlegup = np.mean(X[:, :, left_leg_up, :], axis=2)
		x_leftlegdown = np.mean(X[:, :, left_leg_down, :], axis=2)
		x_rightlegup = np.mean(X[:, :, right_leg_up, :], axis=2)
		x_rightlegdown = np.mean(X[:, :, right_leg_down, :], axis=2)
		x_head_1 = np.mean(X[:, :, head_1, :], axis=2)
		x_head_2 = np.mean(X[:, :, head_2, :], axis=2)
		x_leftarmup = np.mean(X[:, :, left_arm_up, :], axis=2)
		x_leftarmdown_1 = np.mean(X[:, :, left_arm_down_1, :], axis=2)
		x_leftarmdown_2 = np.mean(X[:, :, left_arm_down_2, :], axis=2)
		x_leftarmdown_3 = np.mean(X[:, :, left_arm_down_3, :], axis=2)
		x_rightarmup = np.mean(X[:, :, right_arm_up, :], axis=2)
		x_rightarmdown_1 = np.mean(X[:, :, right_arm_down_1, :], axis=2)
		x_rightarmdown_2 = np.mean(X[:, :, right_arm_down_2, :], axis=2)
		x_rightarmdown_3 = np.mean(X[:, :, right_arm_down_3, :], axis=2)
		shoulder_1 = np.mean(X[:, :, shoulder_1, :], axis=2)
		shoulder_2 = np.mean(X[:, :, shoulder_2, :], axis=2)
		elbow_1 = np.mean(X[:, :, elbow_1, :], axis=2)
		elbow_2 = np.mean(X[:, :, elbow_2, :], axis=2)
		spine_mm = np.mean(X[:, :, spine_mm, :], axis=2)
		hip_1 = np.mean(X[:, :, hip_1, :], axis=2)
		hip_2 = np.mean(X[:, :, hip_2, :], axis=2)
		knee_1 = np.mean(X[:, :, knee_1, :], axis=2)
		knee_2 = np.mean(X[:, :, knee_2, :], axis=2)
		X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup,
		                         x_rightlegdown, x_torso, x_head_1, x_head_2, x_leftarmup,
		                         x_leftarmdown_1, x_leftarmdown_2, x_leftarmdown_3,
		                         x_rightarmup, x_rightarmdown_1, x_rightarmdown_2, x_rightarmdown_3,
		                         shoulder_1, shoulder_2, elbow_1, elbow_2, spine_mm,
		                         hip_1, hip_2, knee_1, knee_2), axis=-1) \
			.reshape([X.shape[0], X.shape[1], 24, 3])
		# 25+24
		X_interp = np.concatenate((X, X_part), axis=-2)
	# Changed
	elif joint_num == 20:
		# left_leg_up = [12, 13]
		# left_leg_down = [14, 15]
		# right_leg_up = [16, 17]
		# right_leg_down = [18, 19]
		# torso = [0, 1]
		# head = [2, 3]
		# left_arm_up = [4, 5]
		# left_arm_down = [6, 7]
		# right_arm_up = [8, 9]
		# right_arm_down = [10, 11]
		#
		left_leg_up = [12, 13]
		left_leg_down = [14, 15]
		right_leg_up = [16, 17]
		right_leg_down = [18, 19]
		torso = [0, 1]
		head_1 = [2, 3]
		# head_2 = [2, 20]
		left_arm_up = [4, 5]
		left_arm_down_1 = [6, 7]
		# left_arm_down_2 = [7, 21]
		# left_arm_down_3 = [6, 22]
		right_arm_up = [8, 9]
		right_arm_down_1 = [10, 11]
		# right_arm_down_2 = [11, 23]
		# right_arm_down_3 = [10, 24]
		# shoulder_1 = [4, 20]
		# shoulder_2 = [8, 20]
		shoulder_1 = [4, 2]
		shoulder_2 = [8, 2]
		elbow_1 = [5, 6]
		elbow_2 = [9, 10]
		# spine_mm = [20, 1]
		spine_mm = [2, 1]
		hip_1 = [0, 12]
		hip_2 = [0, 16]
		knee_1 = [13, 14]
		knee_2 = [17, 18]

		x_torso = np.mean(X[:, :, torso, :], axis=2)  # [N * T, V=1]
		x_leftlegup = np.mean(X[:, :, left_leg_up, :], axis=2)
		x_leftlegdown = np.mean(X[:, :, left_leg_down, :], axis=2)
		x_rightlegup = np.mean(X[:, :, right_leg_up, :], axis=2)
		x_rightlegdown = np.mean(X[:, :, right_leg_down, :], axis=2)
		x_head_1 = np.mean(X[:, :, head_1, :], axis=2)
		# x_head_2 = np.mean(X[:, :, head_2, :], axis=2)
		x_leftarmup = np.mean(X[:, :, left_arm_up, :], axis=2)
		x_leftarmdown_1 = np.mean(X[:, :, left_arm_down_1, :], axis=2)
		# x_leftarmdown_2 = np.mean(X[:, :, left_arm_down_2, :], axis=2)
		# x_leftarmdown_3 = np.mean(X[:, :, left_arm_down_3, :], axis=2)
		x_rightarmup = np.mean(X[:, :, right_arm_up, :], axis=2)
		x_rightarmdown_1 = np.mean(X[:, :, right_arm_down_1, :], axis=2)
		# x_rightarmdown_2 = np.mean(X[:, :, right_arm_down_2, :], axis=2)
		# x_rightarmdown_3 = np.mean(X[:, :, right_arm_down_3, :], axis=2)
		shoulder_1 = np.mean(X[:, :, shoulder_1, :], axis=2)
		shoulder_2 = np.mean(X[:, :, shoulder_2, :], axis=2)
		elbow_1 = np.mean(X[:, :, elbow_1, :], axis=2)
		elbow_2 = np.mean(X[:, :, elbow_2, :], axis=2)
		spine_mm = np.mean(X[:, :, spine_mm, :], axis=2)
		hip_1 = np.mean(X[:, :, hip_1, :], axis=2)
		hip_2 = np.mean(X[:, :, hip_2, :], axis=2)
		knee_1 = np.mean(X[:, :, knee_1, :], axis=2)
		knee_2 = np.mean(X[:, :, knee_2, :], axis=2)
		X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup,
		                         x_rightlegdown, x_torso, x_head_1, x_leftarmup,
		                         x_leftarmdown_1,
		                         x_rightarmup, x_rightarmdown_1,
		                         shoulder_1, shoulder_2, elbow_1, elbow_2, spine_mm,
		                         hip_1, hip_2, knee_1, knee_2), axis=-1) \
			.reshape([X.shape[0], X.shape[1], 19, 3])
		# 25+24
		X_interp = np.concatenate((X, X_part), axis=-2)
	return X_interp

def generate_denser_graph_data(X, adj, joint_num=20):
	adj_temp = copy.deepcopy(adj)
	adj_temp = adj_temp.tolist()
	node_num = len(adj_temp)
	cnt = node_num
	for i in range(node_num):
		for j in range(node_num):
			if adj_temp[i][j] == 1:
				adj_temp[i][j] = adj_temp[j][i] = 0
				new_node = np.mean(X[:, :, [i, j], :], axis=2)
				# print(new_node.shape)
				if cnt == node_num:
					X_interp = new_node
				else:
					X_interp = np.concatenate((X_interp, new_node), axis=-1)
					# print(X_interp.shape)
					# print(i, j)
				# print(i, j, cnt)
				cnt += 1
	# print(X_interp.shape)
	X_interp = np.reshape(X_interp, [X.shape[0], X.shape[1], node_num-1, 3])
	X_interp = np.concatenate((X, X_interp), axis=-2)
	return X_interp

"""
 Calculate normalized area under curves.
"""
def cal_nAUC(scores, labels):
	scores = np.array(scores)
	labels = np.array(labels)
	# Compute micro-average ROC curve and ROC area
	fpr, tpr, thresholds = roc_curve(labels.ravel(), scores.ravel())
	roc_auc = auc(fpr, tpr)
	return roc_auc

"""
 Generate training data with evenly distributed classes.
"""
# replaced by random shuffling
def class_samp_gen(X, y, ids_, batch_size):
	class_num = len(ids_.keys())
	ids_ = sorted(ids_.items(), key=lambda item: item[0])
	cnt = 0
	all_batch_X = []
	all_batch_y = []
	total = y.shape[0]
	batch_num = total // batch_size * 2
	batch_num = total // batch_size * 2
	class_in_bacth = class_num
	batch_per_class = batch_size // class_in_bacth
	class_cnt = class_in_bacth
	# print(total, batch_num, batch_per_class)
	for i in range(batch_num):
		batch_X = []
		batch_y = []
		for k, v in ids_[class_cnt-class_in_bacth:class_cnt]:
			# print(k, len(v))
			# cnt += len(v)
			if len(v[batch_per_class*i:batch_per_class*(i+1)]) < batch_per_class:
				rand_ind = np.random.choice(len(v), batch_per_class)
				v_array = np.array(v)
				samp_per_class = v_array[rand_ind].tolist()
				batch_X.extend(samp_per_class)
			else:
				batch_X.extend(v[batch_per_class*i:batch_per_class*(i+1)])
			batch_y.extend(batch_per_class * [k])
		if class_cnt + class_in_bacth > class_num and class_cnt <= class_num:
			class_cnt = class_num
		else:
			class_cnt = class_cnt + class_in_bacth
		all_batch_X.extend(batch_X)
		all_batch_y.extend(batch_y)
	# print(len(all_batch_X), len(all_batch_y))
	X_train = X[all_batch_X]
	y_train = np.array(all_batch_y)
	return X_train, y_train

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
	nb_graphs = adj.shape[0]
	mt = np.empty(adj.shape)
	for g in range(nb_graphs):
		mt[g] = np.eye(adj.shape[1])
		for _ in range(nhood):
			mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
		for i in range(sizes[g]):
			for j in range(sizes[g]):
				if mt[g][i][j] > 0.0:
					mt[g][i][j] = 1.0
	return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
	"""Parse index file."""
	index = []
	for line in open(filename):
		index.append(int(line.strip()))
	return index


def sample_mask(idx, l):
	"""Create mask."""
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
	"""Load data."""
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				objects.append(pkl.load(f))

	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
	test_idx_range = np.sort(test_idx_reorder)

	if dataset_str == 'citeseer':
		# Fix citeseer dataset (there are some isolated nodes in the graph)
		# Find isolated nodes, add them as zero-vecs into the right position
		test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
		tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
		tx_extended[test_idx_range - min(test_idx_range), :] = tx
		tx = tx_extended
		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range - min(test_idx_range), :] = ty
		ty = ty_extended

	features = sp.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]

	idx_test = test_idx_range.tolist()
	idx_train = range(len(y))
	idx_val = range(len(y), len(y) + 500)

	train_mask = sample_mask(idx_train, labels.shape[0])
	val_mask = sample_mask(idx_val, labels.shape[0])
	test_mask = sample_mask(idx_test, labels.shape[0])

	y_train = np.zeros(labels.shape)
	y_val = np.zeros(labels.shape)
	y_test = np.zeros(labels.shape)
	y_train[train_mask, :] = labels[train_mask, :]
	y_val[val_mask, :] = labels[val_mask, :]
	y_test[test_mask, :] = labels[test_mask, :]

	# print(adj.shape)
	# print(features.shape)

	return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_random_data(size):
	adj = sp.random(size, size, density=0.002)  # density similar to cora
	features = sp.random(size, 1000, density=0.015)
	int_labels = np.random.randint(7, size=(size))
	labels = np.zeros((size, 7))  # Nx7
	labels[np.arange(size), int_labels] = 1

	train_mask = np.zeros((size,)).astype(bool)
	train_mask[np.arange(size)[0:int(size / 2)]] = 1

	val_mask = np.zeros((size,)).astype(bool)
	val_mask[np.arange(size)[int(size / 2):]] = 1

	test_mask = np.zeros((size,)).astype(bool)
	test_mask[np.arange(size)[int(size / 2):]] = 1

	y_train = np.zeros(labels.shape)
	y_val = np.zeros(labels.shape)
	y_test = np.zeros(labels.shape)
	y_train[train_mask, :] = labels[train_mask, :]
	y_val[val_mask, :] = labels[val_mask, :]
	y_test[test_mask, :] = labels[test_mask, :]

	# sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
	return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
	"""Convert sparse matrix to tuple representation."""

	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)

	return sparse_mx


def standardize_data(f, train_mask):
	"""Standardize feature matrix and convert to tuple representation"""
	# standardize data
	f = f.todense()
	mu = f[train_mask == True, :].mean(axis=0)
	sigma = f[train_mask == True, :].std(axis=0)
	f = f[:, np.squeeze(np.array(sigma > 0))]
	mu = f[train_mask == True, :].mean(axis=0)
	sigma = f[train_mask == True, :].std(axis=0)
	f = (f - mu) / sigma
	return f


def preprocess_features(features):
	"""Row-normalize feature matrix and convert to tuple representation"""
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)
	return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix."""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
	num_nodes = adj.shape[0]
	adj = adj + sp.eye(num_nodes)  # self-loop
	adj[adj > 0.0] = 1.0
	if not sp.isspmatrix_coo(adj):
		adj = adj.tocoo()
	adj = adj.astype(np.float32)
	indices = np.vstack(
		(adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
	# return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
	return indices, adj.data, adj.shape
