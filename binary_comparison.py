import tensorflow as tf
import util
import numpy as np
import h5py
import scipy
import modelzoo
import os
import json

def binary_to_profile(binary_model_dir,profile_data_dir):
    model = tf.keras.models.load_model(binary_model_dir,compile=True)
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer(index=-2).output)
    testset = util.make_dataset(profile_data_dir, 'test', util.load_stats(profile_data_dir), batch_size=128)
    json_path = os.path.join(profile_data_dir, 'statistics.json')
    with open(json_path) as json_file:
        params = json.load(json_file)

    target_list = []
    pred_list = []
    for i, (x, y) in enumerate(testset):
        target_cov = np.average(y.numpy(),axis = 1)
        pred_cov = model.predict(x)
        target_list.append(target_cov)
        pred_list.append(pred_cov)

    target = np.concatenate(target_list)
    pred = np.concatenate(pred_list)
    r_list = []
    for i in range(0,15):
        r_list.append(scipy.stats.pearsonr(target[:,i],pred[:,i])[0])

    return r_list

def profile_to_binary(run_dir,binary_data_dir):
    model = modelzoo.load_model(run_dir,False)
    f = h5py.File(binary_data_dir,'r')
    test_x = f['x_test'][()]
    test_y = f['y_test'][()]
    f.close()

    pred_profile = model.predict(test_x)
    pred_cov = np.sum(pred_profile,axis=1)

    exp_num = pred_cov.shape[-1]

    p_profile = []
    f_profile = []
    for exp in range(0,exp_num):
        exp_pred_cov = pred_cov[:,exp]
        exp_target_label = test_y[:,exp]

        peak_idx = np.nonzero(exp_target_label)
        flat_idx = np.where(exp_target_label == 0)

        peak_cov = exp_pred_cov[peak_idx]
        flat_cov = exp_pred_cov[flat_idx]

        p_profile.append(np.array(peak_cov))
        f_profile.append(np.array(flat_cov))

    return p_profile,f_profile
