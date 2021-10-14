import tensorflow as tf
import h5py
import explain
import custom_fit
import modelzoo
import os,json
import util
import time
import sys
import numpy as np
from tfr_evaluate import get_run_metadata

output_directory = util.make_dir('augmentation_robustness_output')

model_path = sys.argv[1]
id = model_path.split('-')[-1]

model = modelzoo.load_model(model_path, compile=True)
metadata = get_run_metadata(model_path)

# overlapping_testset_path = '/home/shush/profile/QuantPred/datasets/step1K_chr8_thresh2/i_3072_w_1/'
# testset = util.make_dataset(overlapping_testset_path, 'test', util.load_stats(overlapping_testset_path),
#                             batch_size=512, shuffle=False)
# X_test, Y_test = util.convert_tfr_to_np(testset)
#
# h5f = h5py.File('/home/shush/profile/QuantPred/datasets/step1K_chr8_thresh2/testset.h5', 'w')
# h5f.create_dataset('X', data=X_test)
# h5f.create_dataset('Y', data=Y_test)
# h5f.close()

h5f = h5py.File('./datasets/step1K_chr8_thresh2/step1K_chr8_thresh2.h5', 'r')
X = h5f['X'][:]
Y = h5f['Y'][:]
start_time = time.time()
variance_saliency, variance_pred = explain.batch_robustness_test(X[:107,:,:], Y[:107,:,:], model,
                                                                batch_size=10,
                                                                visualize=False,
                                                                shift_num=10)
print(np.mean(variance_pred))
print("--- %s seconds ---" % (time.time() - start_time))

tf.keras.backend.clear_session()

metadata['average predictions variance'] = np.mean(variance_pred)
metadata['average saliency variance'] = np.mean(variance_saliency)

print(metadata)
metadata.to_csv(os.path.join(output_directory, id.strip('/')+'.csv'))
