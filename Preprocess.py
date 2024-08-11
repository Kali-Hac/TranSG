import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import copy
clip_num = 10
overlap = 20
time_steps = 20

def preprocess_IAS(type, clip_num=10):
	cnt = 0
	p_txt = []
	p = []
	if type == 'A':
		name = 'TestingA'
	elif type == 'B':
		name = 'TestingB'
	elif type == 'Train':
		name = 'Training'
		p_gal = []
		p_gal_txt = []
	print('(IAS - %s) txt -> npy' % name)
	all_person_folder = os.listdir('./IAS/' + name)
	for a, b, c in os.walk('./IAS/' + name):
		if len(b) == 0:
			txt_t = []
			for file in c:
				if 'skel.txt' in file:
					txt_t.append(a + '/' + file)
			txt_t.sort()
			# print(len(txt_t))
			if type == 'Train':
				# for train random half
				# perm = np.random.permutation(len(txt_t))
				# txt_t = np.array(txt_t)[perm].tolist()
				# p_txt.append(txt_t[:len(txt_t)//2])
				p_txt.append(txt_t)
				p.append(cnt)
				# for gallery
				# p_gal_txt.append(txt_t[len(txt_t) // 2:])
				# p_gal.append(cnt)
			else:
				p_txt.append(txt_t)
				p.append(cnt)
			cnt += 1
	assert len(p_txt) == len(p)
	X = []
	Y = []
	Z = []
	f_num = 0
	abnormal_num = 0
	normal_num = 0
	for index, txts_dir in enumerate(p_txt):
		# print(len(txts_dir))
		# print(index)
		# print(txts_dir)
		p_x = []
		p_y = []
		p_z = []
		for txt in txts_dir:
			track_state = True
			with open(txt, 'r') as f:
				lines = f.readlines()
				label_t = []
				x_t = []
				y_t = []
				z_t = []
				for line in lines:
					temp = line.split(',')
					TrackingState = float(temp[6])
					QualityFlage = float(temp[7])
					if TrackingState == 0:
						track_state = False
						break
					label_t.append(float(temp[0]))
					x_t.append(float(temp[1]))
					y_t.append(float(temp[2]))
					z_t.append(float(temp[3]))
			if not track_state:
				abnormal_num += 1
				continue
			else:
				normal_num += 1
			p_x.append(x_t)
			p_y.append(y_t)
			p_z.append(z_t)
		# print(len(p_x))
		assert len(p_x) == len(p_y) and len(p_y) == len(p_z)
		# print(len(p_x[clip_num:-clip_num]))
		if type != 'Train':
			f_num += len(p_x[clip_num:-clip_num])
			X.append(p_x[clip_num:-clip_num])
			Y.append(p_y[clip_num:-clip_num])
			Z.append(p_z[clip_num:-clip_num])
		else:
			f_num += len(p_x[clip_num:])
			X.append(p_x[clip_num:])
			Y.append(p_y[clip_num:])
			Z.append(p_z[clip_num:])

	# print()
	# print(str(normal_num / len(set(p))))
	# print(str(f_num / len(set(p))))
	# print(abnormal_num)
	# print(normal_num)
	np.save('./IAS/' + name + '_label.npy', p)
	np.save('./IAS/' + name + '_x.npy', X)
	np.save('./IAS/' + name + '_y.npy', Y)
	np.save('./IAS/' + name + '_z.npy', Z)
	# print(X)
	# print(np.array(X).shape)
	if type == 'Train':
		X = []
		Y = []
		Z = []
		f_num = 0
		abnormal_num = 0
		normal_num = 0
		for index, txts_dir in enumerate(p_gal_txt):
			# print(index)
			# print(txts_dir)
			p_x = []
			p_y = []
			p_z = []
			for txt in txts_dir:
				track_state = True
				with open(txt, 'r') as f:
					lines = f.readlines()
					label_t = []
					x_t = []
					y_t = []
					z_t = []
					for line in lines:
						temp = line.split(',')
						TrackingState = float(temp[6])
						QualityFlage = float(temp[7])
						if TrackingState == 0:
							track_state = False
							break
						label_t.append(float(temp[0]))
						x_t.append(float(temp[1]))
						y_t.append(float(temp[2]))
						z_t.append(float(temp[3]))
				if not track_state:
					abnormal_num += 1
					continue
				else:
					normal_num += 1
				p_x.append(x_t)
				p_y.append(y_t)
				p_z.append(z_t)
			assert len(p_x) == len(p_y) and len(p_y) == len(p_z)
			# print(len(p_x[clip_num:-clip_num]))
			# f_num += len(p_x[clip_num:-clip_num])
			X.append(p_x[:-clip_num])
			# print(len(X), len(X[0]))
			Y.append(p_y[:-clip_num])
			Z.append(p_z[:-clip_num])

		# print()
		# print(str(normal_num / len(set(p))))
		# print(str(f_num / len(set(p))))
		# print(abnormal_num)
		# print(normal_num)
		np.save('./IAS/Gallery_label.npy', p_gal)
		np.save('./IAS/Gallery_x.npy', X)
		np.save('./IAS/Gallery_y.npy', Y)
		np.save('./IAS/Gallery_z.npy', Z)
		# print(np.array(X).shape)


def preprocess_BIWI(type, clip_num=10):
	cnt = 0
	p = []
	p_txt = []
	if type == 'W':
		name = 'Testing/Walking'
	elif type == 'S':
		name = 'Testing/Still'
	elif type == 'Train':
		name = 'Training'
	print('(BIWI - %s) txt -> npy' % name)
	for a, b, c in os.walk('./BIWI/' + name):
		if len(b) == 0:
			txt_t = []
			for file in c:
				if 'skel.txt' in file:
					txt_t.append(a + '/' + file)
			txt_t.sort()
			p_txt.append(txt_t)
			p_num = int(a.split('/')[-1])
			# print(p_num)
			p.append(p_num)
			cnt += 1
	assert len(p_txt) == len(p)
	X = []
	Y = []
	Z = []
	f_num = 0
	abnormal_num = 0
	normal_num = 0
	for index, txts_dir in enumerate(p_txt):
		# print(index)
		# print(txts_dir)
		p_x = []
		p_y = []
		p_z = []
		for txt in txts_dir:
			track_state = True
			with open(txt, 'r') as f:
				lines = f.readlines()
				label_t = []
				x_t = []
				y_t = []
				z_t = []
				for index, line in enumerate(lines[:20]):
					temp = line.split(',')
					TrackingState = float(temp[6])
					QualityFlage = float(temp[7])
					if TrackingState == 0 and index < 20:
						track_state = False
						break
					label_t.append(float(temp[0]))
					x_t.append(float(temp[1]))
					y_t.append(float(temp[2]))
					z_t.append(float(temp[3]))
					# if QualityFlage > 0:
					# 	print(line)
			if not track_state:
				abnormal_num += 1
				continue
			else:
				normal_num += 1
			p_x.append(x_t)
			p_y.append(y_t)
			p_z.append(z_t)
		assert len(p_x) == len(p_y) and len(p_y) == len(p_z)
		# print(len(p_x[clip_num:-clip_num]))
		f_num += len(p_x[clip_num:-clip_num])
		X.append(p_x[clip_num:-clip_num])
		Y.append(p_y[clip_num:-clip_num])
		Z.append(p_z[clip_num:-clip_num])
	# print(p)
	# print()
	# print(str(normal_num / len(set(p))))
	# print(str(f_num / len(set(p))))
	# print(abnormal_num)
	# print(normal_num)
	if type == 'W':
		np.save('./BIWI/Walking_label.npy', p)
		np.save('./BIWI/Walking_x.npy', X)
		np.save('./BIWI/Walking_y.npy', Y)
		np.save('./BIWI/Walking_z.npy', Z)
	elif type == 'S':
		np.save('./BIWI/Still_label.npy', p)
		np.save('./BIWI/Still_x.npy', X)
		np.save('./BIWI/Still_y.npy', Y)
		np.save('./BIWI/Still_z.npy', Z)
	else:
		np.save('./BIWI/Train_label.npy', p)
		np.save('./BIWI/Train_x.npy', X)
		np.save('./BIWI/Train_y.npy', Y)
		np.save('./BIWI/Train_z.npy', Z)



# preprocess KGBD dataset
def process_dataset_KGBD(save_dir, fr=6):
	global overlap, time_steps
	try:
		os.mkdir('Datasets/KGBD/')
	except:
		pass
	save_dir = 'Datasets/KGBD/' + save_dir
	try:
		os.mkdir(save_dir)
	except:
		pass
	overlap = fr
	time_steps = fr
	a = 'KGBD/kinect gait raw dataset/'
	p_dir = os.listdir(a)
	label = []
	X = []
	Y = []
	Z = []
	gal_label = []
	gal_X = []
	gal_Y = []
	gal_Z = []
	pro_label = []
	pro_X = []
	pro_Y = []
	pro_Z = []
	for p in p_dir:
		if 'Person' not in p:
			continue
		# print(p)
		p_num = int(p.split('Person')[1])
		data_dir = a + p
		fold_dir = os.listdir(data_dir)
		fold_dir = sorted(fold_dir)
		if p_num == 15:
			# only 3 txt, for train (1) + probe (1) + gallery (1)
			rand_seq_nums = np.random.choice(len(fold_dir), 2, replace=False)
			gal_seq_num = rand_seq_nums[:1]
			pro_seq_num = rand_seq_nums[1]
			print('Person ID: %d | Gallery Seqs: %d | Probe Seq: %d | Total Seq: %d' % (
			p_num, gal_seq_num[0], pro_seq_num, len(fold_dir)))
		else:
			rand_seq_nums = np.random.choice(len(fold_dir), 3, replace=False)
			gal_seq_num = rand_seq_nums[:2]
			pro_seq_num = rand_seq_nums[2]
			print('Person ID: %d | Gallery Seqs: %d, %d| Probe Seq: %d | Total Seq: %d' % (p_num, gal_seq_num[0], gal_seq_num[1], pro_seq_num, len(fold_dir)))
		for seq_num, txt_ind in enumerate(fold_dir):
			with open(a + p + '/' + txt_ind, 'r') as f:
				lines = f.readlines()
				x = []
				y = []
				z = []
				x_t = []
				y_t = []
				z_t = []
				cnt = 0
				for line in lines:
					coors = line.split(';')[1:]
					x_t.append(float(coors[0]))
					y_t.append(float(coors[1]))
					z_t.append(float(coors[2]))
					cnt += 1
					if cnt % 20 == 0:
						x.append(x_t)
						y.append(y_t)
						z.append(z_t)
						x_t = []
						y_t = []
						z_t = []
			if seq_num not in rand_seq_nums:
				X.append(x[clip_num:-clip_num])
				Y.append(y[clip_num:-clip_num])
				Z.append(z[clip_num:-clip_num])
				label.append(p_num)
			elif seq_num in gal_seq_num:
				gal_X.append(x[clip_num:-clip_num])
				gal_Y.append(y[clip_num:-clip_num])
				gal_Z.append(z[clip_num:-clip_num])
				gal_label.append(p_num)
			elif seq_num == pro_seq_num:
				pro_X.append(x[clip_num:-clip_num])
				pro_Y.append(y[clip_num:-clip_num])
				pro_Z.append(z[clip_num:-clip_num])
				pro_label.append(p_num)

	def get_npy_data(save_dir, X, Y ,Z, label, type):
		x_source = []
		y_source = []
		z_source = []
		# x_target = []
		# y_target = []
		# z_target = []
		frames_cnt = 0
		ids = {}
		frame_id = []
		for s in range(0, 1):
			for index, xx_row in enumerate(X):
				# for frame in x_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				x_row = copy.deepcopy(xx_row[s:])
				if type=='XX':
					t = [x_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(x_row) - overlap) // (overlap // 2) + 1)]
				else:
					t = [x_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(x_row) // overlap)]
				person = label[index]
				if person not in ids.keys():
					ids[person] = []
				if len(t) > 0:
					ids[person].extend([i + frames_cnt for i in range(len(t))])
				frame_id.extend([person for i in range(len(t))])
				frames_cnt += len(t)
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					x_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	x_target.extend(rev_t)
			for index, yy_row in enumerate(Y):
				# for frame in y_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				y_row = yy_row[s:]
				if type == 'XX':
					t = [y_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(y_row) - overlap) // (overlap // 2) + 1)]
				else:
					t = [y_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(y_row) // overlap)]
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					y_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	y_target.extend(rev_t)
			for index, zz_row in enumerate(Z):
				# for frame in z_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				z_row = zz_row[s:]
				if type == 'XX':
					t = [z_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(z_row) - overlap) // (overlap // 2) + 1)]
				else:
					t = [z_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(z_row) // overlap)]
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					z_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	z_target.extend(rev_t)
		x_source = np.array(x_source)
		y_source = np.array(y_source)
		z_source = np.array(z_source)
		# x_target = np.array(x_target)
		# y_target = np.array(y_target)
		# z_target = np.array(z_target)
		# x_data -= x_data[:, 0]
		# y_data -= y_data[:, 0]
		# z_data -= z_data[:, 0]
		assert len(x) == len(y) and len(y) == len(z)
		# for i in range(1, 165):
		# 	if i not in ids.keys():
		# 		print('lack: ', i)
		if type == 'Train':
			save_dir += 'train_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 'source_x_KGBD_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 'source_y_KGBD_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 'source_z_KGBD_' + str(fr) + '.npy', z_source)
			# np.save(save_dir + 'target_x_KGBD_' + str(fr) + '.npy', x_target)
			# np.save(save_dir + 'target_y_KGBD_' + str(fr) + '.npy', y_target)
			# np.save(save_dir + 'target_z_KGBD_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_KGBD_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_KGBD_' + str(fr) + '.npy', frame_id)
		else:
			save_dir += 'test_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			save_dir += type + '/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 't_source_x_KGBD_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 't_source_y_KGBD_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 't_source_z_KGBD_' + str(fr) + '.npy', z_source)
		# 	np.save(save_dir + 't_target_x_KGBD_' + str(fr) + '.npy', x_target)
		# 	np.save(save_dir + 't_target_y_KGBD_' + str(fr) + '.npy', y_target)
		# 	np.save(save_dir + 't_target_z_KGBD_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_KGBD_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_KGBD_' + str(fr) + '.npy', frame_id)

	get_npy_data(save_dir, X, Y, Z, label, type='Train')
	get_npy_data(save_dir, gal_X, gal_Y, gal_Z, gal_label, type='gallery')
	get_npy_data(save_dir, pro_X, pro_Y, pro_Z, pro_label, type='probe')


# preprocess IAS dataset
def process_dataset_IAS(save_dir, fr=6):
	preprocess_IAS('Train')
	preprocess_IAS('A')
	preprocess_IAS('B')
	try:
		os.mkdir('Datasets/IAS/')
	except:
		pass
	save_dir = 'Datasets/IAS/' + save_dir
	try:
		os.mkdir(save_dir)
	except:
		pass
	global overlap, time_steps
	overlap = fr
	time_steps = fr
	X = np.load('./IAS/Training_x.npy').tolist()
	Y = np.load('./IAS/Training_y.npy').tolist()
	Z = np.load('./IAS/Training_z.npy').tolist()
	label = np.load('./IAS/Training_label.npy').tolist()

	# gal_X = np.load('./IAS/Gallery_x.npy').tolist()
	# gal_Y = np.load('./IAS/Gallery_y.npy').tolist()
	# gal_Z = np.load('./IAS/Gallery_z.npy').tolist()
	# gal_label = np.load('./IAS/Gallery_label.npy').tolist()

	A_X = np.load('./IAS/TestingA_x.npy').tolist()
	A_Y = np.load('./IAS/TestingA_y.npy').tolist()
	A_Z = np.load('./IAS/TestingA_z.npy').tolist()
	A_label = np.load('./IAS/TestingA_label.npy').tolist()

	B_X = np.load('./IAS/TestingB_x.npy').tolist()
	B_Y = np.load('./IAS/TestingB_y.npy').tolist()
	B_Z = np.load('./IAS/TestingB_z.npy').tolist()
	B_label = np.load('./IAS/TestingB_label.npy').tolist()

	# print(len(X), len(X[0]))
	# exit()
	# X = X.tolist()
	# Y = Y.tolist()
	# Z = Z.tolist()
	# label = label.tolist()
	def get_npy_data(save_dir, X, Y ,Z, label, type):
		print('Process %s data' % type)
		x_source = []
		y_source = []
		z_source = []
		# x_target = []
		# y_target = []
		# z_target = []
		frames_cnt = 0
		ids = {}
		frame_id = []
		for s in range(0, time_steps):
			for index, xx_row in enumerate(X):
				# for frame in x_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				x_row = copy.deepcopy(xx_row[s:])
				# print(type, 'x_row', x_row)
				if type=='Train':
					t = [x_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(x_row) - overlap) // (overlap // 2) + 1)]
				elif s == 0:
					t = [x_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(x_row) // overlap)]
				else:
					continue
				person = label[index]
				if person not in ids.keys():
					ids[person] = []
				if len(t) > 0:
					ids[person].extend([i + frames_cnt for i in range(len(t))])
				frame_id.extend([person for i in range(len(t))])
				frames_cnt += len(t)
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				# print('ori', t)
				if len(t) > 0:
					t = sum(t, [])
					x_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	x_target.extend(rev_t)
			for index, yy_row in enumerate(Y):
				# for frame in y_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				y_row = yy_row[s:]
				if type == 'Train':
					t = [y_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(y_row) - overlap) // (overlap // 2) + 1)]
				elif s==0:
					t = [y_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(y_row) // overlap)]
				else:
					continue
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					y_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	y_target.extend(rev_t)
			for index, zz_row in enumerate(Z):
				# for frame in z_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				z_row = zz_row[s:]
				if type == 'Train':
					t = [z_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(z_row) - overlap) // (overlap // 2) + 1)]
				elif s == 0:
					t = [z_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(z_row) // overlap)]
				else:
					continue
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					z_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	z_target.extend(rev_t)
		x_source = np.array(x_source)
		y_source = np.array(y_source)
		z_source = np.array(z_source)
		# x_target = np.array(x_target)
		# y_target = np.array(y_target)
		# z_target = np.array(z_target)
		# x_data -= x_data[:, 0]
		# y_data -= y_data[:, 0]
		# z_data -= z_data[:, 0]
		# assert len(x) == len(y) and len(y) == len(z)
		# for i in range(0, 11):
		# 	if i not in ids.keys():
		# 		print('lack: ', type, i)
		if type == 'Train':
			save_dir += 'train_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 'source_x_IAS_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 'source_y_IAS_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 'source_z_IAS_' + str(fr) + '.npy', z_source)
			# np.save(save_dir + 'target_x_IAS_' + str(fr) + '.npy', x_target)
			# np.save(save_dir + 'target_y_IAS_' + str(fr) + '.npy', y_target)
			# np.save(save_dir + 'target_z_IAS_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_IAS_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_IAS_' + str(fr) + '.npy', frame_id)
		else:
			save_dir += 'test_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			save_dir += type + '/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 't_source_x_IAS_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 't_source_y_IAS_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 't_source_z_IAS_' + str(fr) + '.npy', z_source)
			# np.save(save_dir + 't_target_x_IAS-A_' + str(fr) + '.npy', x_target)
			# np.save(save_dir + 't_target_y_IAS-A_' + str(fr) + '.npy', y_target)
			# np.save(save_dir + 't_target_z_IAS-A_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_IAS_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_IAS_' + str(fr) + '.npy', frame_id)

	get_npy_data(save_dir, X, Y, Z, label, type='Train')
	# get_npy_data(save_dir, gal_X, gal_Y, gal_Z, gal_label, type='gallery')
	get_npy_data(save_dir, A_X, A_Y, A_Z, A_label, type='A')
	get_npy_data(save_dir, B_X, B_Y, B_Z, B_label, type='B')



# preprocess BIWI dataset
def process_dataset_BIWI(save_dir, fr=6):
	preprocess_BIWI('Train')
	preprocess_BIWI('W')
	preprocess_BIWI('S')
	try:
		os.mkdir('Datasets/BIWI/')
	except:
		pass
	save_dir = 'Datasets/BIWI/' + save_dir
	try:
		os.mkdir(save_dir)
	except:
		pass
	global overlap, time_steps
	overlap = fr
	time_steps = fr
	X = np.load('./BIWI/Train_x.npy')
	Y = np.load('./BIWI/Train_y.npy')
	Z = np.load('./BIWI/Train_z.npy')
	label = np.load('./BIWI/Train_label.npy')

	W_X = np.load('./BIWI/Walking_x.npy').tolist()
	W_Y = np.load('./BIWI/Walking_y.npy').tolist()
	W_Z = np.load('./BIWI/Walking_z.npy').tolist()
	W_label = np.load('./BIWI/Walking_label.npy').tolist()

	S_X = np.load('./BIWI/Still_x.npy').tolist()
	S_Y = np.load('./BIWI/Still_y.npy').tolist()
	S_Z = np.load('./BIWI/Still_z.npy').tolist()
	S_label = np.load('./BIWI/Still_label.npy').tolist()


	def get_npy_data(save_dir, X, Y ,Z, label, type):
		print('Process %s data' % type)
		x_source = []
		y_source = []
		z_source = []
		# x_target = []
		# y_target = []
		# z_target = []
		frames_cnt = 0
		ids = {}
		frame_id = []
		for s in range(0, time_steps):
			for index, xx_row in enumerate(X):
				# for frame in x_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				x_row = copy.deepcopy(xx_row[s:])
				if type=='Train':
					t = [x_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(x_row) - overlap) // (overlap // 2) + 1)]
				elif s==0:
					t = [x_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(x_row) // overlap)]
				else:
					continue
				person = label[index]
				if person not in ids.keys():
					ids[person] = []
				if len(t) > 0:
					ids[person].extend([i + frames_cnt for i in range(len(t))])
				frame_id.extend([person for i in range(len(t))])
				frames_cnt += len(t)
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					x_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	x_target.extend(rev_t)
			for index, yy_row in enumerate(Y):
				# for frame in y_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				y_row = yy_row[s:]
				if type == 'Train':
					t = [y_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(y_row) - overlap) // (overlap // 2) + 1)]
				elif s==0:
					t = [y_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(y_row) // overlap)]
				else:
					continue
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					y_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	y_target.extend(rev_t)
			for index, zz_row in enumerate(Z):
				# for frame in z_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				z_row = zz_row[s:]
				if type == 'Train':
					t = [z_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(z_row) - overlap) // (overlap // 2) + 1)]
				elif s==0:
					t = [z_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(z_row) // overlap)]
				else:
					continue
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					z_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	z_target.extend(rev_t)
		x_source = np.array(x_source)
		y_source = np.array(y_source)
		z_source = np.array(z_source)
		# x_target = np.array(x_target)
		# y_target = np.array(y_target)
		# z_target = np.array(z_target)
		# x_data -= x_data[:, 0]
		# y_data -= y_data[:, 0]
		# z_data -= z_data[:, 0]
		# assert len(x) == len(y) and len(y) == len(z)
		if type == 'Train':
			save_dir += 'train_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 'source_x_BIWI_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 'source_y_BIWI_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 'source_z_BIWI_' + str(fr) + '.npy', z_source)
			# np.save(save_dir + 'target_x_IAS_' + str(fr) + '.npy', x_target)
			# np.save(save_dir + 'target_y_IAS_' + str(fr) + '.npy', y_target)
			# np.save(save_dir + 'target_z_IAS_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_BIWI_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_BIWI_' + str(fr) + '.npy', frame_id)
		else:
			save_dir += 'test_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			save_dir += type + '/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 't_source_x_BIWI_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 't_source_y_BIWI_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 't_source_z_BIWI_' + str(fr) + '.npy', z_source)
			# np.save(save_dir + 't_target_x_IAS-A_' + str(fr) + '.npy', x_target)
			# np.save(save_dir + 't_target_y_IAS-A_' + str(fr) + '.npy', y_target)
			# np.save(save_dir + 't_target_z_IAS-A_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_BIWI_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_BIWI_' + str(fr) + '.npy', frame_id)

	get_npy_data(save_dir, X, Y, Z, label, type='Train')
	get_npy_data(save_dir, W_X, W_Y, W_Z, W_label, type='Walking')
	get_npy_data(save_dir, S_X, S_Y, S_Z, S_label, type='Still')


# preprocess KS20 dataset
def process_dataset_KS20(save_dir, fr=10):
	global overlap, time_steps
	try:
		os.mkdir('Datasets/KS20/')
	except:
		pass
	save_dir = 'Datasets/KS20/' + save_dir
	try:
		os.mkdir(save_dir)
	except:
		pass
	overlap = fr
	time_steps = fr
	a = 'KS20/'
	label = []
	X = []
	Y = []
	Z = []
	label_gallery = []
	X_gallery = []
	Y_gallery = []
	Z_gallery = []
	label_probe = []
	X_probe = []
	Y_probe = []
	Z_probe = []
	views = ['frontal', 'left_diagonal', 'left_lateral', 'right_diagonal', 'right_lateral']
	settings = ['gc1', 'gc2', 'gc3']
	# bodies = ['1','2','3','4','7','8','9','11','12','13',
	#          '14','15','16', '17', '19', '20', '21', '22', '23', '24']
	# corresponeding testing views
	gallery_views = []
	probe_views = []
	skeleton_num = 0
	for i in range(20):
		gallery_num, probe_num = np.random.choice(3, 2, replace=False)
		# random_num = np.random.randint(0, 3)
		# test_views.append(settings[random_num])
		gallery_views.append(settings[gallery_num])
		probe_views.append(settings[probe_num])
	for view in views:
		p_dir = os.listdir(a + view)
		current_num = -1
		p_dir = sorted(p_dir)
		for p in p_dir:
			# if 'ren.py' in p:
			# 	continue
			p_num = int(p.split('_')[1].split('body')[1])
			if 1 <= p_num <= 4:
				p_num -= 1
			elif 7 <= p_num <= 9:
				p_num -= 3
			elif 11 <= p_num <= 17:
				p_num -= 4
			elif 19 <= p_num <= 24:
				p_num -= 5
			print('Process ID: %d | Gallery Seq: %s | Probe Seq: %s | Train Seq: Others' % (p_num, gallery_views[current_num], probe_views[current_num]))
			current_setting = p.split('_')[0]
			# 6 not in train or probe
			# if p_num == 6:
			# 	print()
			# 	print(current_setting)
			if current_num != p_num:
				# print(current_num, p_num, current_setting)
				if current_num != -1:
					if current_setting == gallery_views[current_num]:
						# print('gallery', current_setting)
						X_gallery.append(x)
						Y_gallery.append(y)
						Z_gallery.append(z)
						label_gallery.append(current_num)
					elif current_setting == probe_views[current_num]:
						# print('probe', current_setting)
						X_probe.append(x)
						Y_probe.append(y)
						Z_probe.append(z)
						label_probe.append(current_num)
					else:
						# print('train', current_setting)
						X.append(x)
						Y.append(y)
						Z.append(z)
						label.append(current_num)
				current_num = p_num
				x = []
				y = []
				z = []
			timestamp = int(p.split('_')[2])
			with open(a + view + '/' + p, 'r') as f:
				skeleton_num += 1
				lines = f.readlines()
				cnt = 0
				x_t = []
				y_t = []
				z_t = []
				for i in range(25):
					coor_x, coor_y, coor_z = \
						float(lines[i].split(',')[1]), float(lines[i].split(',')[2]), float(lines[i].split(',')[3])
					x_t.append(coor_x)
					y_t.append(coor_y)
					z_t.append(coor_z)
				cnt += 1
			x.append(x_t)
			y.append(y_t)
			z.append(z_t)
	# print(skeleton_num)
	# exit(1)
	def get_npy_data(save_dir, X, Y ,Z, label, type):
		print('Process %s data' % type)
		x_source = []
		y_source = []
		z_source = []
		# x_target = []
		# y_target = []
		# z_target = []
		frames_cnt = 0
		ids = {}
		frame_id = []
		for s in range(0, time_steps):
			for index, xx_row in enumerate(X):
				# for frame in x_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				x_row = copy.deepcopy(xx_row[s:])
				if type=='Train':
					t = [x_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(x_row) - overlap) // (overlap // 2) + 1)]
				elif s==0:
					t = [x_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(x_row) // overlap)]
				else:
					continue
				person = label[index]
				if person not in ids.keys():
					ids[person] = []
				if len(t) > 0:
					ids[person].extend([i + frames_cnt for i in range(len(t))])
				frame_id.extend([person for i in range(len(t))])
				frames_cnt += len(t)
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					x_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	x_target.extend(rev_t)
			for index, yy_row in enumerate(Y):
				# for frame in y_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				y_row = yy_row[s:]
				if type == 'Train':
					t = [y_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(y_row) - overlap) // (overlap // 2) + 1)]
				elif s==0:
					t = [y_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(y_row) // overlap)]
				else:
					continue
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					y_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	y_target.extend(rev_t)
			for index, zz_row in enumerate(Z):
				# for frame in z_row:
				# 	for i in range(1, 25):
				# 		frame[i] -= frame[0]
				z_row = zz_row[s:]
				if type == 'Train':
					t = [z_row[(i * overlap // 2): (i * overlap // 2 + overlap)] for i in
					     range((len(z_row) - overlap) // (overlap // 2) + 1)]
				elif s==0:
					t = [z_row[(i * overlap): (i * overlap + overlap)] for i in
					     range(len(z_row) // overlap)]
				else:
					continue
				# rev_t = copy.deepcopy(t)
				# for k in rev_t:
				# 	k.reverse()
				if len(t) > 0:
					t = sum(t, [])
					z_source.extend(t)
				# 	rev_t = sum(rev_t, [])
				# 	z_target.extend(rev_t)
		x_source = np.array(x_source)
		y_source = np.array(y_source)
		z_source = np.array(z_source)
		# x_target = np.array(x_target)
		# y_target = np.array(y_target)
		# z_target = np.array(z_target)
		# x_data -= x_data[:, 0]
		# y_data -= y_data[:, 0]
		# z_data -= z_data[:, 0]
		# assert len(x) == len(y) and len(y) == len(z)
		# for i in range(0, 20):
		# 	if i not in ids.keys():
		# 		print('lack: ', type, i)
		if type == 'Train':
			save_dir += 'train_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 'source_x_KS20_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 'source_y_KS20_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 'source_z_KS20_' + str(fr) + '.npy', z_source)
			# np.save(save_dir + 'target_x_IAS_' + str(fr) + '.npy', x_target)
			# np.save(save_dir + 'target_y_IAS_' + str(fr) + '.npy', y_target)
			# np.save(save_dir + 'target_z_IAS_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_KS20_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_KS20_' + str(fr) + '.npy', frame_id)
		else:
			save_dir += 'test_npy_data/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			save_dir += type + '/'
			try:
				os.mkdir(save_dir)
			except:
				pass
			# permutation = np.random.permutation(x_source.shape[0])
			# x_source = x_source[permutation,]
			# y_source = y_source[permutation,]
			# z_source = z_source[permutation,]
			# x_target = x_target[permutation,]
			# y_target = y_target[permutation,]
			# z_target = z_target[permutation,]
			# ids = ids[permutation,]
			# frame_id = frame_id[permutation,]
			np.save(save_dir + 't_source_x_KS20_' + str(fr) + '.npy', x_source)
			np.save(save_dir + 't_source_y_KS20_' + str(fr) + '.npy', y_source)
			np.save(save_dir + 't_source_z_KS20_' + str(fr) + '.npy', z_source)
			# np.save(save_dir + 't_target_x_IAS-A_' + str(fr) + '.npy', x_target)
			# np.save(save_dir + 't_target_y_IAS-A_' + str(fr) + '.npy', y_target)
			# np.save(save_dir + 't_target_z_IAS-A_' + str(fr) + '.npy', z_target)
			np.save(save_dir + 'ids_KS20_' + str(fr) + '.npy', ids)
			np.save(save_dir + 'frame_id_KS20_' + str(fr) + '.npy', frame_id)

	get_npy_data(save_dir, X, Y, Z, label, type='Train')
	get_npy_data(save_dir, X_gallery, Y_gallery, Z_gallery, label_gallery, type='gallery')
	get_npy_data(save_dir, X_probe, Y_probe, Z_probe, label_probe, type='probe')


if __name__ == '__main__':
	import sys
	f = sys.argv[1]
	f = int(f)
	try:
		os.mkdir('Datasets')
	except:
		pass
	process_dataset_KGBD(str(f) + '/', f)
	process_dataset_IAS(str(f) + '/', f)
	process_dataset_BIWI(str(f) + '/', f)
	process_dataset_KS20(str(f) + '/', f)