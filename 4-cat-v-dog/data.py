import os, shutil

base_dir = os.join(os.getcwd(), 'data')
original_dataset_dir = os.path.join(base_dir, 'original_data', 'train')

data_segment_names = ['train', 'validation', 'test']
labels = ['cat', 'dog']

data_segments = {
	data_segment_names[0]: (0, 1000),
	data_segment_names[1]: (1000, 1500),
	data_segment_names[2]: (1500, 2000)
}

def mkdir(dir):
	try:
		os.mkdir(dir)
	except OSError:
		print('{} not created - alread exists'.format(dir), dir)

def prepare_data():
	# Make data segments directories
	for segment in data_segment_names:
		segment_dir = os.path.join(base_dir, segment)
		mkdir(segment_dir)
		# Make label directories for each data segment
		for label in labels:
			label_segment_dir = os.path.join(segment_dir, label + 's')
			mkdir(label_segment_dir)
			# Copy data over for each data segment-label pair
			fnames = [(label+'.{}.jpg').format(i) for i in range(data_segments[segment][0], data_segments[segment][1])]
			for fname in fnames:
				src = os.path.join(original_dataset_dir, fname)
				dst = os.path.join(label_segment_dir, fname)
				shutil.copyfile(src, dst)

if __name__=='__main__':
	prepare_data()