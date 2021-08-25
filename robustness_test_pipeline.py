import tensorflow as tf
import h5py
import explain
import custom_fit
import modelzoo
from loss import *
import os,json
import util
import pandas as pd
import importlib
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import metrics
import csv

test_data = './datasets/VCF/filtered_test.h5'
f =  h5py.File(test_data, "r")
x = f['x_test'][()]
y = f['y_test'][()]
f.close()

model_path_list = ['/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_153757-5y918d62',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_153800-cc03oavi',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_153800-tf9gu91e',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_153809-h0ysow8l',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_153809-zwjgoagu',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_153813-kb86iu5s',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_225231-w70zz5ir',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_225520-4jmi917g',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_225734-h6wcurcm',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_230120-nowa6cgv',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_230319-nybdfnwn',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210805_231021-m2wdkllq',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_052808-840bvndx',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_053425-fwb7rxmn',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_053639-2134rbdd',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_054505-6c6hgr24',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_055130-v88hk7ul',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_060124-qmtkpvst',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_093447-64aa6v5i',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_093447-gb65vmpm',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_093535-qvn1krx4',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_155248-k8nyz1lp',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_155249-ehlni9bp',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_160011-x08ybxia',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_160216-r52ymhnl',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_161840-s6ygwerq',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_163237-jakn6rh5',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_201706-7aogjqwy',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_202204-xzsdk1so',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210806_203302-uka056um',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210807_065001-rah3yiv6',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210807_070106-4hge39qr',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210807_072612-nujcuu0o',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210807_161143-trvtey2h',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210807_162504-ds0go4h0',
       '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210807_170154-qlqlpzy4']

saliency_file = open('./datasets/robustness_saliency_eval.csv','w', newline = '\n')
saliencywriter = csv.writer(saliency_file, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

pred_file = open('./datasets/robustness_pred_eval.csv','w', newline = '\n')
predwriter = csv.writer(pred_file, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

for model_path in model_path_list:
    model = modelzoo.load_model(model_path,compile=True)
    var_saliency,var_pred = explain.robustness_test(x,y,
                                                  model,visualize = False,shift_num =10)

    saliencywriter.writerow(var_saliency)
    predwriter.writerow(var_pred)



    tf.keras.backend.clear_session()
