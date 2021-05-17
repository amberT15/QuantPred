import numpy as np
import tensorflow as tf
from tensorflow import keras

def select_top_pred(pred,num_task):
    
    task_top_list = []
    for i in range(0,num_task):
        task_profile = pred[:,:,i]
        task_mean =np.squeeze(np.mean(task_profile,axis = 1))
        task_index = task_mean.argsort()[-2:]
        task_top_list.append(task_index)
    task_top_list = np.array(task_top_list)
    return task_top_list



def peak_saliency_map(X, model, class_index, func=tf.math.reduce_mean):
    """fast function to generate saliency maps"""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        pred = model(X)

        peak_index = tf.math.argmax(pred[:,:,class_index],axis=1)
        batch_indices = []
        for i in range(0,X.shape[0]):
            column_indices = tf.range(peak_index[i]-25,peak_index[i]+25,dtype='int32')
            row_indices = tf.keras.backend.repeat_elements(tf.constant([i]),50, axis=0)
            full_indices = tf.stack([row_indices, column_indices], axis=1)
            batch_indices.append([full_indices])
            outputs = tf.keras.backend.mean(tf.gather_nd(pred[:,:,class_index],batch_indices),axis=2)

        return tape.gradient(outputs, X)
    