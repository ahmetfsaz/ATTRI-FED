'''
Created on Oct 24, 2021
Modified on May 03, 2022

@author: yashas_saidutta\
@modified_by: AFS
'''
import multiprocessing
import time
import sys
import os
import shutil
import time as tm
import numpy as np
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from datetime import datetime



if not tuple(map(int, (tf.__version__.split(".")))) <= tuple(map(int, ("1.15.4".split(".")))):
    del tf
    import tensorflow.compat.v1 as tf

    tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_v2_behavior()
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import platform
import copy
from pprint import pprint


global n_latent, beta1_AD, num_epochs_MI_U_and_Yhat, loss_sat_lim, client_dict
n_latent = 50
beta1_AD = 1.5
beta2_AD = 1.5
indep_AD = 1.5
num_epochs_MI_U_and_Yhat = 1000
loss_sat_lim = 5
client_dict = {}
plot_counter = 0

global num_epochs_adversarial_training_enc_dec
num_epochs_adversarial_training_enc_dec = 10000 # 250
# LOOK AT WHAT THE HELL IS LAYER NORMALIZATION AND WHY DO YOU USE IT, ALSO WATCH FOR VANISHING GRADIENTS!
# CHANGE MAE WITH KL DIVERGENCE
# IT DOES NOT MAKE SENSE TO USE JUST THE LAST ADVERSARY, ENCODER, DECODER WHATSOEVER! SET IT TO VALIDATION CRITERIA SINCE YOU ALREADY KNOW LOSS IS THE MOST IMP. THING!
num_epochs_adversarial_training_adv = 1000  # 100
num_transfer_init_enc_dec_epochs = 1000
num_transfer_init_adv2_epochs = 1000
num_transfer_init_adv_epochs = 1000  # 250
num_collective_init_adv_epochs = 1000
num_global_FL_epochs = 50
num_local_FL_epochs = 50

# net_dict is network dictionary, 'call' is the function you use to evaluate a given input using the FCNN
class FCNN(tf.keras.Model):
    def __init__(self, hidden_neuron_list, output_dim, inp_shp):
        super(FCNN, self).__init__()
        self.net_dict = {}
        self.layer_dict = {}
        self.output_dim = output_dim
        self.hidden_neuron_list = hidden_neuron_list
        self.curr_layer_num = 0
        self.inp_shp = inp_shp

        for indx, hidden_neuron_num in enumerate(self.hidden_neuron_list):
            self.layer_dict['layer{}'.format(self.curr_layer_num)] = tf.keras.layers.Dense(units=hidden_neuron_num,
                                                                                           activation='relu',
                                                                                           kernel_initializer='he_uniform',
                                                                                           name='Layer{}'.format(
                                                                                               self.curr_layer_num))
            self.curr_layer_num += 1

            # LAYER NORMALIZATION
            self.layer_dict['layer{}'.format(self.curr_layer_num)] = tf.keras.layers.LayerNormalization()
            self.curr_layer_num += 1
        self.layer_dict['layer{}'.format(self.curr_layer_num)] = tf.keras.layers.Dense(units=output_dim,
                                                                                       activation=None,
                                                                                       kernel_initializer='he_uniform',
                                                                                       name='Layer{}'.format(
                                                                                           self.curr_layer_num))
        self.curr_layer_num += 1

    def call(self, input):
        for layer_num in range(self.curr_layer_num):
            if layer_num == 0:
                self.net_dict['Op_layer{}'.format(layer_num)] = self.layer_dict['layer{}'.format(layer_num)](input)
                # tf.keras.Input(shape=(self.inp_shp,))
            else:
                self.net_dict['Op_layer{}'.format(layer_num)] = self.layer_dict['layer{}'.format(layer_num)](
                    self.net_dict['Op_layer{}'.format(layer_num - 1)])
        return self.net_dict['Op_layer{}'.format(self.curr_layer_num - 1)]

    def get_layer_dict(self):
        return self.layer_dict

class EdgeNode():

    def __init__(self, train_data, val_data, test_data, cl_ID, sess, NUM_CLASS_TGT, NUM_CLASS_PRIVATE, x, u, v, t, u_TFL, u_int,
                 v_int, t_int, compressed, learning_rate, tf_warmup_wt, adv_beta, num_tr_datapoints, num_va_datapoints, num_te_datapoints,
                 batch_size, whole_adult_test_data, etrain_data, eval_data, etest_data, enum_tr_datapoints,
                 enum_va_datapoints, enum_te_datapoints, directory, mnist_shape, u_one_hot_rep, adv_beta2, indep_beta):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.net_dict = {}
        self.clientID = cl_ID
        self.clientID_str = str(cl_ID)
        self.sess = sess
        self.NUM_CLASS_TGT = NUM_CLASS_TGT
        self.NUM_CLASS_PRIVATE = NUM_CLASS_PRIVATE
        self.x = x
        self.u = u
        self.v = v
        self.t = t
        self.u_TFL = u_TFL
        self.u_int = u_int
        self.v_int = v_int
        self.t_int = t_int
        self.compressed = compressed
        self.learning_rate = learning_rate
        self.tf_warmup_wt = tf_warmup_wt
        self.adv_beta = adv_beta
        self.num_tr_datapoints = num_tr_datapoints
        self.num_va_datapoints = num_va_datapoints
        self.num_te_datapoints = num_te_datapoints
        self.train_step_GVIB_and_ADV = {}
        self.train_step_GVIB_only = {}
        self.train_step_ADV_only = {}
        self.train_step_Enc_Dec = {}
        self.post_hoc = {}
        self.train_step_MI_U_and_Yhat_only = {}
        self.batch_size = batch_size
        self.encoder = {}
        self.decoder = {}
        self.adversary = {}
        self.adversary2 = {}
        self.instance_classifier = {}
        self.whole_adult_test_data = whole_adult_test_data
        self.tr_compressed = []
        self.val_compressed = []
        self.te_compressed = []
        self.etrain_data = etrain_data
        self.eval_data = eval_data
        self.etest_data = etest_data
        self.enum_tr_datapoints = enum_tr_datapoints
        self.enum_va_datapoints = enum_va_datapoints
        self.enum_te_datapoints = enum_te_datapoints
        self.train_combined = []
        self.val_combined = []
        self.test_combined = []
        self.train_y_combined = []
        self.val_y_combined = []
        self.test_y_combined = []
        self.logits_LR_classifier_tr = []
        self.logits_LR_classifier_val = []
        self.logits_LR_classifier_test = []
        self.etrain_data_d = copy.deepcopy(etrain_data)
        self.eval_data_d = copy.deepcopy(eval_data)
        self.etest_data_d = copy.deepcopy(etest_data)
        self.directory = directory
        self.mnist_shape = mnist_shape
        self.u_one_hot_rep = u_one_hot_rep
        self.adv_beta2 = adv_beta2
        self.indep_beta = indep_beta
        self.plot_counter = 0


    def construct_model(self):
        with tf.variable_scope(self.clientID_str):
            with tf.variable_scope('GVIB'):
                with tf.variable_scope('Encoder_latent_IB'):
                    EncoderNN_latent_IB = FCNN([1024, 1024], n_latent, self.mnist_shape)
                    self.encoder = EncoderNN_latent_IB
                    self.net_dict['y_hat'] = EncoderNN_latent_IB(self.x)

                with tf.variable_scope('Decoder_predict_target'):
                    DecoderNN_pred_tgt = FCNN([32], self.NUM_CLASS_TGT, n_latent)
                    self.decoder = DecoderNN_pred_tgt
                    self.net_dict['tgt_logits'] = DecoderNN_pred_tgt(self.net_dict['y_hat'])
                    # Total number of bits divided by total number of pixels.
                    self.net_dict['bpp'] = 0

            with tf.variable_scope('Adversarial'):
                with tf.variable_scope('Predict_private_wt_yhat'):
                    DecoderNN_pred_private_wt_yhat = FCNN([1024, 32], self.NUM_CLASS_PRIVATE, n_latent)
                    self.adversary = DecoderNN_pred_private_wt_yhat
                    self.net_dict['private_logits_wt_yhat'] = DecoderNN_pred_private_wt_yhat(self.net_dict['y_hat'])

            with tf.variable_scope('Reweighting'):
                with tf.variable_scope('Log_reg_classifier'):
                    Log_Reg_Clssfr = FCNN([1024, 1024, 256], 2, self.mnist_shape)
                    self.instance_classifier = Log_Reg_Clssfr
                    self.net_dict['instance_prob'] = Log_Reg_Clssfr(self.x)

            with tf.variable_scope('Adversarial2'):
                with tf.variable_scope('Predict_target_wt_yhat_u'):
                    adversarial_yashas = FCNN([1024, 32], self.NUM_CLASS_TGT, n_latent + 1)
                    self.adversary2 = adversarial_yashas
                    self.net_dict['concatted'] = tf.concat([self.net_dict['y_hat'], self.u_one_hot_rep], axis=1)
                    self.net_dict['target_wt_yhat_u'] = adversarial_yashas(self.net_dict['concatted'])

            # train another classifier to see privacy performance after encoders and decoders are fully trained, metric
            with tf.variable_scope('Post_hoc_MI'):
                with tf.variable_scope('MI_U_and_Yhat_only'):
                    self.post_hoc = FCNN([1024, 1024, 1024, 256, 32], self.NUM_CLASS_PRIVATE, n_latent)
                    self.net_dict['private_logits_given_yhat'] = self.post_hoc(self.compressed)

            with tf.variable_scope('GVIB_Loss'):
                self.net_dict['Loss_CE'] = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.v_int,
                                                                   logits=self.net_dict[
                                                                       'tgt_logits']
                                                                   ))

                self.net_dict['Loss_P1_U_and_yhat_AD'] = -tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.u_int,
                                                                   logits=self.net_dict['private_logits_wt_yhat']
                                                                   ))

                self.net_dict['Loss_adv2'] = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.v_int,
                                                                   logits=self.net_dict[
                                                                       'target_wt_yhat_u']
                                                                   ))

                self.net_dict['Loss_log_reg'] = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.t_int,
                                                                   logits=self.net_dict['instance_prob']
                                                                   ))

                mae = tf.keras.losses.MeanAbsoluteError()
                self.net_dict['Loss_Collective'] = -mae(self.u_TFL, self.net_dict['private_logits_wt_yhat'])
                # log_division = tf.math.log(tf.math.divide(self.net_dict['private_logits_wt_yhat'], self.u_TFL))
                # log_division = tf.math.log(tf.math.divide(self.u_TFL,self.net_dict['private_logits_wt_yhat']))
                # self.net_dict['Loss_Collective'] = -tf.reduce_sum(tf.math.multiply(self.net_dict['private_logits_wt_yhat'], log_division))
                # self.net_dict['Loss_Collective'] = -tf.reduce_sum(tf.math.multiply(self.u_TFL, log_division))


                # So, here is the sign of adversarial loss correct???? MAYBE NOT?? -- CORRECT, Loss is defined MINUS above!!!
                # self.net_dict['Loss_GVIB'] = (1 + self.tf_warmup_wt) * self.net_dict['Loss_CE'] + \
                #                              self.tf_warmup_wt * self.adv_beta * self.net_dict['Loss_P1_U_and_yhat_AD']

                self.net_dict['Loss_GVIB'] = (1 + self.tf_warmup_wt) * self.net_dict['Loss_CE'] + \
                                             self.tf_warmup_wt * self.adv_beta * self.net_dict['Loss_P1_U_and_yhat_AD'] + \
                                             self.tf_warmup_wt * self.adv_beta2 * self.net_dict['Loss_adv2'] + \
                                             self.tf_warmup_wt * (self.adv_beta2 + self.indep_beta) * self.net_dict['bpp']

                with tf.variable_scope('Accuracy'):
                    self.net_dict['tgt_prob'] = tf.nn.softmax(self.net_dict['tgt_logits'], axis=1)
                    self.net_dict['tgt_predict'] = tf.math.argmax(self.net_dict['tgt_prob'], axis=1,
                                                                  output_type=tf.int32)
                    equality = tf.math.equal(self.net_dict['tgt_predict'], self.v_int)
                    self.net_dict['tgt_acc'] = tf.reduce_mean(tf.cast(equality, tf.float32)) * 100.0

                with tf.variable_scope('Accuracy_adv'):
                    self.net_dict['priv_prob'] = tf.nn.softmax(self.net_dict['private_logits_wt_yhat'], axis=1)
                    self.net_dict['priv_predict'] = tf.math.argmax(self.net_dict['priv_prob'], axis=1,
                                                                  output_type=tf.int32)
                    equality = tf.math.equal(self.net_dict['priv_predict'], self.u_int)
                    self.net_dict['priv_acc'] = tf.reduce_mean(tf.cast(equality, tf.float32)) * 100.0

                with tf.variable_scope('Accuracy_adv2'):
                    self.net_dict['adv2_prob'] = tf.nn.softmax(self.net_dict['target_wt_yhat_u'], axis=1)
                    self.net_dict['adv2_predict'] = tf.math.argmax(self.net_dict['adv2_prob'], axis=1,
                                                                  output_type=tf.int32)
                    equality = tf.math.equal(self.net_dict['adv2_predict'], self.v_int)
                    self.net_dict['adv2_acc'] = tf.reduce_mean(tf.cast(equality, tf.float32)) * 100.0

                with tf.variable_scope('Accuracy_LR'):
                    self.net_dict['ins_prob'] = tf.nn.softmax(self.net_dict['instance_prob'], axis=1)
                    self.net_dict['ins_predict'] = tf.math.argmax(self.net_dict['ins_prob'], axis=1,
                                                                  output_type=tf.int32)
                    equality = tf.math.equal(self.net_dict['ins_predict'], self.t_int)
                    self.net_dict['log_ins_acc'] = tf.reduce_mean(tf.cast(equality, tf.float32)) * 100.0

            with tf.variable_scope('Post_hoc_MI_Loss'):
                with tf.variable_scope('MI_U_and_Yhat_only'):
                    self.net_dict['Loss_MI_U_and_Yhat_only'] = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.u_int,
                                                                       logits=self.net_dict[
                                                                           'private_logits_given_yhat']))
                    with tf.variable_scope('Accuracy'):
                        self.net_dict['private_prob_given_yhat'] = tf.nn.softmax(
                            self.net_dict['private_logits_given_yhat'], axis=1)
                        self.net_dict['private_predict_given_yhat'] = tf.math.argmax(
                            self.net_dict['private_prob_given_yhat'], axis=1,
                            output_type=tf.int32)
                        equality = tf.math.equal(self.net_dict['private_predict_given_yhat'], self.u_int)
                        self.net_dict['priavte_acc_P1'] = tf.reduce_mean(tf.cast(equality, tf.float32)) * 100.0

            # determines the set of variables where gradients will be applied
            sc_GV_enc_only = self.clientID_str + '/GVIB/Encoder_latent_IB'
            sc_GV_dec_only = self.clientID_str + '/GVIB/Decoder_predict_target'
            sc_GV = self.clientID_str + '/GVIB'
            sc_LR = self.clientID_str + '/Reweighting'
            sc_adv = self.clientID_str + '/Adversarial'
            sc_adv2 = self.clientID_str + '/Adversarial2'
            sc_ph = self.clientID_str + '/Post_hoc_MI/MI_U_and_Yhat_only'
            vars_GVIB_enc_only = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc_GV_enc_only)
            vars_GVIB_dec_only = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc_GV_dec_only)
            vars_GVIB = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc_GV)
            vars_LR = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc_LR)
            vars_ADV = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc_adv)
            vars_ADV2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc_adv2)
            vars_MI_U_and_Yhat = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc_ph)


            with tf.variable_scope('Optimizers'):
                with tf.variable_scope('GVIB_enc_only_optimizer'):
                    optimizer_GVIB_dec_only = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_GVIB_enc_only = optimizer_GVIB_dec_only.minimize(loss=self.net_dict['Loss_GVIB'],
                                                                        var_list=vars_GVIB_enc_only)

                with tf.variable_scope('GVIB_dec_only_optimizer'):
                    optimizer_GVIB_dec_only = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_GVIB_dec_only = optimizer_GVIB_dec_only.minimize(loss=self.net_dict['Loss_CE'],
                                                                        var_list=vars_GVIB_dec_only)

                with tf.variable_scope('Enc_Dec_optimizer'):
                    optimizer_Enc_Dec = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_Enc_Dec = optimizer_Enc_Dec.minimize(loss=self.net_dict['Loss_CE'],
                                                                        var_list=vars_GVIB)

                with tf.variable_scope('Log_Reg_optimizer'):
                    optimizer_LR = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_LR = optimizer_LR.minimize(loss=self.net_dict['Loss_log_reg'],
                                                                        var_list=vars_LR)

                with tf.variable_scope('GVIB_optimizer'):
                    optimizer_GVIB = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_GVIB_only = optimizer_GVIB.minimize(loss=self.net_dict['Loss_GVIB'],
                                                                        var_list=vars_GVIB)

                with tf.variable_scope('ADV_optimizer'):
                    optimizer_ADV = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_ADV_only = optimizer_ADV.minimize(loss=-self.net_dict['Loss_P1_U_and_yhat_AD'],
                                                                      var_list=vars_ADV)\

                with tf.variable_scope('ADV2_optimizer'):
                    optimizer_ADV2 = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_ADV2_only = optimizer_ADV2.minimize(loss=self.net_dict['Loss_adv2'],
                                                                      var_list=vars_ADV2)

                with tf.variable_scope('ADV_Reg_optimizer'):
                    optimizer_ADV_reg = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_ADV_reg = optimizer_ADV_reg.minimize(loss=-self.net_dict['Loss_Collective'],
                                                                      var_list=vars_ADV)

                with tf.variable_scope('ADV_and_ENC_optimizer'):
                    optimizer_ADV_ENC = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4)
                    self.train_step_ADV_ENC = optimizer_ADV_ENC.minimize(loss=-self.net_dict['Loss_P1_U_and_yhat_AD'],
                                                                      var_list=vars_GVIB_enc_only+vars_ADV)

                self.train_step_GVIB_and_ADV = tf.group(self.train_step_ADV_only, self.train_step_GVIB_only)

                with tf.variable_scope('MI_U_and_Yhat_only_optimizer'):
                    optimizer_MI_U_and_Yhat_only = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999,
                                                                          epsilon=1e-4)
                    self.train_step_MI_U_and_Yhat_only = optimizer_MI_U_and_Yhat_only.minimize(
                        loss=self.net_dict['Loss_MI_U_and_Yhat_only'], var_list=vars_MI_U_and_Yhat)

        print('Model ' + self.clientID_str + ' constructed')

    def training_fnc(self, train_data, batch_size, num_tr_datapoints, learning_rate_py, py_warmup_weight, tr_dict,
                     epoch_num, print_true, string_name, beta=beta1_AD, collective_y=np.zeros((6000, 10))):

        num_tr_iters = num_tr_datapoints // batch_size

        if epoch_num >= -1:
            py_warmup_weight = np.clip(py_warmup_weight * 1.5, 1e-3, 1)  # was py_warmup_weight * 1.5, 1e-3, 1.0
        if epoch_num >= 0:
            tr_tgt_acc = 0.0
            tr_loss_CE = 0.0
            tr_loss_P1_U_and_yhat_AD = 0.0
            tr_loss_GVIB = 0.0
            acc_adv = 0.0
            self.current_predictions_adv = []
            self.current_predictions = []
            tr_time = tm.time()
            for tr_iter in range(num_tr_iters):
                tr_batch = train_data[tr_iter * batch_size:(tr_iter + 1) * batch_size, :]
                col_batch = collective_y[tr_iter * batch_size:(tr_iter + 1) * batch_size, :]
                x_batch = tr_batch[:, 0:-2]
                priv_batch = tr_batch[:, -2]
                tgt_batch = tr_batch[:, -1]

                feed_dict = {self.x: x_batch,
                             self.u: priv_batch,
                             self.v: tgt_batch,
                             self.u_TFL: col_batch,
                             self.learning_rate: learning_rate_py,
                             self.tf_warmup_wt: py_warmup_weight,
                             self.adv_beta: beta}
                # print('*' * 60)
                # print('*' * 60)
                # print('*' * 60)
                # print(self.sess.run(self.u_TFL, feed_dict=feed_dict))
                # print('#' * 60)
                # print('#' * 60)
                # print('#' * 60)
                # print(self.sess.run(self.net_dict['private_logits_wt_yhat'],
                #                     feed_dict=feed_dict))
                # print('D' * 60)
                # print('D' * 60)
                # print('D' * 60)
                tr_op_dict = self.sess.run(tr_dict, feed_dict=feed_dict)
                tr_tgt_acc += tr_op_dict[1] / float(num_tr_iters)
                tr_loss_CE += tr_op_dict[2] / float(num_tr_iters)
                tr_loss_P1_U_and_yhat_AD += tr_op_dict[3] / float(num_tr_iters)
                tr_loss_GVIB += tr_op_dict[4] / float(num_tr_iters)
                self.current_predictions.append(tr_op_dict[5])
                self.current_predictions_adv.append(tr_op_dict[6])
                acc_adv += tr_op_dict[7] / float(num_tr_iters)

            # self.current_predictions_adv = np.array(self.current_predictions_adv).reshape(len(self.current_predictions_adv)*batch_size, self.NUM_CLASS_PRIVATE)
            self.current_predictions_adv = np.array(self.current_predictions_adv).reshape(len(self.current_predictions_adv) * batch_size, 1)
            self.current_predictions_adv = np.squeeze(self.current_predictions_adv)
            priv_out = self.train_data[0:(num_tr_iters) * batch_size, :]
            tgt_out = priv_out[:, -1]
            priv_out = priv_out[:, -2]
            # rocauc_score_tr = roc_auc_score(priv_out, self.current_predictions_adv)
            rocauc_score_tr = 0

            # self.current_predictions = np.array(self.current_predictions).reshape(len(self.current_predictions) * batch_size, 1)
            # self.current_predictions = np.squeeze(self.current_predictions)
            # print('TRAINING')
            # print(classification_report(tgt_out, self.current_predictions))

            if print_true:
                print(string_name, end='')
                print(
                    'Client %d EPOCH %d - TR - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, P1-AD: %3.5g, ROC_AUC: %3.5g, '
                    'ADV_ACC: %3.5g' % (self.clientID,
                                        epoch_num,
                                        tr_tgt_acc,
                                        tr_loss_GVIB,
                                        tr_loss_CE,
                                        -tr_loss_P1_U_and_yhat_AD,
                                        rocauc_score_tr,
                                        acc_adv))

            tr_time = tm.time() - tr_time

        return tr_tgt_acc, acc_adv, tr_loss_CE, -tr_loss_P1_U_and_yhat_AD, py_warmup_weight, tr_time

    def val_test_func(self, val_data, test_data, batch_size, num_va_datapoints, num_te_datapoints, learning_rate_py,
                      py_warmup_weight, te_va_dict, epoch_num, print_val, print_test, string_name, beta=beta1_AD,
                      collective_y_vl=np.zeros((1000, 10)), collective_y_test=np.zeros((1000, 10))):

        num_va_iters = num_va_datapoints // batch_size
        num_te_iters = num_te_datapoints // batch_size

        va_time = tm.time()
        va_tgt_acc = 0.0
        va_loss_CE = 0.0
        va_loss_P1_U_and_yhat_AD = 0.0
        va_loss_GVIB = 0.0
        va_acc_adv = 0.0
        self.current_predictions_adv = []
        self.current_predictions = []
        priv_out_va = []
        tgt_out_va = []

        val_data_for_epoch = val_data[0:batch_size * num_va_iters, :]
        for va_iter in range(num_va_iters):
            va_batch = val_data_for_epoch[va_iter * batch_size:(va_iter + 1) * batch_size, :]
            col_y_batch = collective_y_vl[va_iter * batch_size:(va_iter + 1) * batch_size, :]
            x_batch = va_batch[:, 0:-2]
            priv_batch = va_batch[:, -2]
            tgt_batch = va_batch[:, -1]
            feed_dict = {self.x: x_batch,
                         self.u: priv_batch,
                         self.v: tgt_batch,
                         self.u_TFL: col_y_batch,
                         self.learning_rate: learning_rate_py,
                         self.tf_warmup_wt: py_warmup_weight,
                         self.adv_beta: beta}
            va_op_dict = self.sess.run(te_va_dict, feed_dict=feed_dict)
            va_tgt_acc += va_op_dict[0] / (float(num_va_iters))
            va_loss_CE += va_op_dict[1] / (float(num_va_iters))
            va_loss_P1_U_and_yhat_AD += va_op_dict[2] / (float(num_va_iters))
            va_loss_GVIB += va_op_dict[3] / (float(num_va_iters))
            self.current_predictions.append(va_op_dict[4])
            self.current_predictions_adv.append(va_op_dict[5])
            va_acc_adv += va_op_dict[6] / float(num_va_iters)
            priv_out_va.append(priv_batch)
            tgt_out_va.append(tgt_batch)

        # self.current_predictions_adv = np.array(self.current_predictions_adv).reshape(len(self.current_predictions_adv) * batch_size, self.NUM_CLASS_PRIVATE)
        self.current_predictions_adv = np.array(self.current_predictions_adv).reshape(
            len(self.current_predictions_adv) * batch_size, 1)
        self.current_predictions_adv = np.squeeze(self.current_predictions_adv)
        priv_out_va = np.array(priv_out_va).reshape(len(priv_out_va) * len(priv_out_va[0]))
        priv_out_va = priv_out_va.astype(int)
        # rocauc_score_va = roc_auc_score(priv_out_va, self.current_predictions_adv)
        rocauc_score_va = 0

        # self.current_predictions = np.array(self.current_predictions).reshape(len(self.current_predictions) * batch_size, 1)
        # self.current_predictions = np.squeeze(self.current_predictions)
        # tgt_out_va = np.array(tgt_out_va).reshape(len(tgt_out_va) * len(tgt_out_va[0]))
        # tgt_out_va = tgt_out_va.astype(int)
        # print('VALIDATION')
        # print(classification_report(tgt_out_va, self.current_predictions))

        if print_val:
            print(string_name, end='')
            print(
                'Client %d EPOCH %d - VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, P1-AD: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                self.clientID,
                epoch_num,
                va_tgt_acc,
                va_loss_GVIB,
                va_loss_CE,
                -va_loss_P1_U_and_yhat_AD,
                rocauc_score_va,
                va_acc_adv))
        va_time = tm.time() - va_time

        te_time = tm.time()
        te_tgt_acc = 0.0
        te_loss_CE = 0.0
        te_loss_P1_U_and_yhat_AD = 0.0
        te_loss_GVIB = 0.0
        te_acc_adv = 0.0
        self.current_predictions_adv = []
        self.current_predictions = []
        priv_out_te = []
        tgt_out_te = []

        test_data_for_epoch = test_data[0:batch_size * num_te_iters, :]

        for te_iter in range(num_te_iters):
            te_batch = test_data_for_epoch[te_iter * batch_size:(te_iter + 1) * batch_size, :]
            x_batch = te_batch[:, 0:-2]
            priv_batch = te_batch[:, -2]
            tgt_batch = te_batch[:, -1]
            col_y_batch = collective_y_test[te_iter * batch_size:(te_iter + 1) * batch_size, :]
            feed_dict = {self.x: x_batch,
                         self.u: priv_batch,
                         self.v: tgt_batch,
                         self.u_TFL: col_y_batch,
                         self.learning_rate: learning_rate_py,
                         self.tf_warmup_wt: py_warmup_weight,
                         self.adv_beta: beta}
            te_op_dict = self.sess.run(te_va_dict, feed_dict=feed_dict)
            te_tgt_acc += te_op_dict[0] / (float(num_te_iters))  # num_te_va_reps_flt *
            te_loss_CE += te_op_dict[1] / (float(num_te_iters))  # num_te_va_reps_flt *
            te_loss_P1_U_and_yhat_AD += te_op_dict[2] / (float(num_te_iters))  # num_te_va_reps_flt *
            te_loss_GVIB += te_op_dict[3] / (float(num_te_iters))  # num_te_va_reps_flt *
            self.current_predictions.append(te_op_dict[4])
            self.current_predictions_adv.append(te_op_dict[5])
            te_acc_adv += te_op_dict[6] / float(num_te_iters)  # * num_te_va_reps)
            priv_out_te.append(priv_batch)
            tgt_out_te.append(tgt_batch)

        # self.current_predictions_adv = np.array(self.current_predictions_adv).reshape(len(self.current_predictions_adv) * batch_size, self.NUM_CLASS_PRIVATE)
        self.current_predictions_adv = np.array(self.current_predictions_adv).reshape(
            len(self.current_predictions_adv) * batch_size, 1)
        self.current_predictions_adv = np.squeeze(self.current_predictions_adv)
        priv_out_te = np.array(priv_out_te).reshape(len(priv_out_te) * len(priv_out_te[0]))
        priv_out_te = priv_out_te.astype(int)
        # rocauc_score_te = roc_auc_score(priv_out_te, self.current_predictions_adv)
        rocauc_score_te = 0

        # self.current_predictions = np.array(self.current_predictions).reshape(len(self.current_predictions) * batch_size, 1)
        # self.current_predictions = np.squeeze(self.current_predictions)
        # tgt_out_te = np.array(tgt_out_te).reshape(len(tgt_out_te) * len(tgt_out_te[0]))
        # tgt_out_te = tgt_out_te.astype(int)
        # print('TESTING')
        # print(classification_report(tgt_out_te, self.current_predictions))

        if print_test:
            print(string_name, end='')
            print(
                'Client %d EPOCH %d - TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, P1-AD: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                self.clientID,
                epoch_num,
                te_tgt_acc,
                te_loss_GVIB,
                te_loss_CE,
                -te_loss_P1_U_and_yhat_AD,
                rocauc_score_te,
                te_acc_adv))

        te_time = tm.time() - te_time


        return va_tgt_acc, va_acc_adv, va_loss_CE, -va_loss_P1_U_and_yhat_AD, va_loss_GVIB, te_tgt_acc, te_acc_adv, \
               te_loss_CE, -te_loss_P1_U_and_yhat_AD, te_loss_GVIB, rocauc_score_va, rocauc_score_te, va_time, te_time


    def init_Log_Reg_Train(self):

        self.combine_EMNIST_and_MNIST()
        batch_size = self.batch_size
        num_tr_iters = (self.train_combined.shape)[0] // batch_size
        num_va_iters = (self.val_combined.shape)[0] // batch_size
        num_te_iters = (self.test_combined.shape)[0] // batch_size

        print('Starting Instance Classification Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        tr_dict = [self.train_step_LR,
                   self.net_dict['log_ins_acc'],
                   self.net_dict['Loss_log_reg'],
                   self.net_dict['ins_predict'],
                   ]

        te_va_dict = [self.net_dict['log_ins_acc'],
                      self.net_dict['Loss_log_reg'],
                      self.net_dict['ins_predict'],
                      ]

        be_va_loss = np.inf
        best_set = False
        be_epoch_num = -100
        loss_sat = 0
        warmup_weight = 1
        learn_rate = 1e-3
        num_epochs = 200

        best_enc_weights = self.get_encoder_weights()
        best_dec_weights = self.get_decoder_weights()

        for epoch_num in range(-1, num_epochs):
            self.combine_EMNIST_and_MNIST()
            if epoch_num >= -1:
                warmup_weight = np.clip(warmup_weight * 1.5, 1e-3, 1)  # was py_warmup_weight * 1.5, 1e-3, 1.0

            if epoch_num >= 0:
                tr_tgt_acc = 0.0
                tr_loss = 0.0
                self.current_predictions = []
                tr_time = tm.time()
                for tr_iter in range(num_tr_iters):
                    tr_batch = self.train_combined[tr_iter * batch_size:(tr_iter + 1) * batch_size, :]
                    x_batch = tr_batch[:, 0:-2]
                    u_batch = tr_batch[:, -2]
                    v_batch = tr_batch[:, -1]
                    tgt_batch = self.train_y_combined[tr_iter * batch_size:(tr_iter + 1) * batch_size]

                    feed_dict = {self.x: x_batch,
                                 self.u: u_batch,
                                 self.v: v_batch,
                                 self.t: tgt_batch,
                                 self.learning_rate: learn_rate,
                                 self.tf_warmup_wt: warmup_weight}
                    tr_op_dict = self.sess.run(tr_dict, feed_dict=feed_dict)
                    tr_tgt_acc += tr_op_dict[1] / float(num_tr_iters)
                    tr_loss += tr_op_dict[2] / float(num_tr_iters)
                    self.current_predictions.append(tr_op_dict[3])


                print('Log Reg Classifier ', end='')
                print(
                    'Client %d EPOCH %d - TR - ACC: %3.5g, Loss: %3.5g' % (self.clientID,
                                                                           epoch_num,
                                                                           tr_tgt_acc,
                                                                           tr_loss))

                tr_time = tm.time() - tr_time

            va_time = tm.time()
            va_tgt_acc = 0.0
            va_loss = 0.0
            self.current_predictions = []

            val_data_for_epoch = self.val_combined[0:batch_size * num_va_iters, :]
            for va_iter in range(num_va_iters):
                va_batch = val_data_for_epoch[va_iter * batch_size:(va_iter + 1) * batch_size, :]
                x_batch = va_batch[:, 0:-2]
                u_batch = va_batch[:, -2]
                v_batch = va_batch[:, -1]
                tgt_batch = self.val_y_combined[va_iter * batch_size:(va_iter + 1) * batch_size]
                feed_dict = {self.x: x_batch,
                             self.u: u_batch,
                             self.v: v_batch,
                             self.t: tgt_batch,
                             self.learning_rate: learn_rate,
                             self.tf_warmup_wt: warmup_weight}
                va_op_dict = self.sess.run(te_va_dict, feed_dict=feed_dict)
                va_tgt_acc += va_op_dict[0] / (float(num_va_iters))
                va_loss += va_op_dict[1] / (float(num_va_iters))
                self.current_predictions.append(va_op_dict[2])

            print('Log Reg Classifier ', end='')
            print(
                'Client %d EPOCH %d - VA - ACC: %3.5g, Loss: %3.5g' % (
                self.clientID,
                epoch_num,
                va_tgt_acc,
                va_loss
                ))
            va_time = tm.time() - va_time

            te_time = tm.time()
            te_tgt_acc = 0.0
            te_loss = 0.0
            self.current_predictions = []

            test_data_for_epoch = self.test_combined[0:batch_size * num_te_iters, :]

            for te_iter in range(num_te_iters):
                te_batch = test_data_for_epoch[te_iter * batch_size:(te_iter + 1) * batch_size, :]
                x_batch = te_batch[:, 0:-2]
                u_batch = te_batch[:, -2]
                v_batch = te_batch[:, -1]
                tgt_batch = self.test_y_combined[te_iter * batch_size:(te_iter + 1) * batch_size]
                feed_dict = {self.x: x_batch,
                             self.u: u_batch,
                             self.v: v_batch,
                             self.t: tgt_batch,
                             self.learning_rate: learn_rate,
                             self.tf_warmup_wt: warmup_weight}
                te_op_dict = self.sess.run(te_va_dict, feed_dict=feed_dict)
                te_tgt_acc += te_op_dict[0] / (float(num_te_iters))  # num_te_va_reps_flt *
                te_loss += te_op_dict[1] / (float(num_te_iters))  # num_te_va_reps_flt *
                self.current_predictions.append(te_op_dict[2])

            print('Log Reg Classifier ', end='')
            print(
                'Client %d EPOCH %d - TE - ACC: %3.5g, Loss: %3.5g' % (
                self.clientID,
                epoch_num,
                te_tgt_acc,
                te_loss))

            te_time = tm.time() - te_time

            if va_loss < be_va_loss and warmup_weight == 1:
                best_set = True
                be_epoch_num = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(va_tgt_acc)
                be_va_loss = copy.deepcopy(va_loss)

                be_te_tgt_acc = copy.deepcopy(te_tgt_acc)
                be_te_loss = copy.deepcopy(te_loss)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if best_set:
                print(
                    'Client %d Log Reg Class Epoch %d - BE VA - ACC: %3.5g, Loss: %3.5g' % (
                        self.clientID,
                        be_epoch_num,
                        be_va_tgt_acc,
                        be_va_loss))

                print(
                    'Client %d Log Reg Class Epoch %d - BE TE - ACC: %3.5g, Loss: %3.5g' % (
                        self.clientID,
                        be_epoch_num,
                        be_te_tgt_acc,
                        be_te_loss))

            if va_loss > 1.01 * be_va_loss:
                loss_sat += 1
            else:
                loss_sat = 0

            if loss_sat == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')

        self.set_encoder_weights(best_enc_weights)
        self.set_decoder_weights(best_dec_weights)

    def transfer_learning_init_EMNIST(self):

        print('Starting EMNIST Classification Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        tr_dict = [self.train_step_Enc_Dec,
                   self.net_dict['tgt_acc'],
                   self.net_dict['Loss_CE'],
                   self.net_dict['Loss_P1_U_and_yhat_AD'],
                   self.net_dict['Loss_GVIB'],
                   self.net_dict['tgt_predict'],
                   self.net_dict['priv_predict'],
                   self.net_dict['priv_acc']
                   ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_CE = np.inf
        best_set_CE = False
        be_epoch_num_CE = -100

        loss_sat_CE = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1
        learn_rate = 1e-3

        for epoch_num in range(-1, num_transfer_init_enc_dec_epochs):

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.etrain_data, batch_size=100,
                                      num_tr_datapoints=self.enum_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='TRL EMNIST ')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.eval_data, test_data=self.etest_data, batch_size=100,
                                   num_va_datapoints=self.enum_va_datapoints, num_te_datapoints=self.enum_te_datapoints,
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='TRL EMNIST ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d TRL EMNIST EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_CE))
            if epoch_num < 0:
                print('Client %d TRL EMNIST EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d TRL EMNIST EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if loss_CE_va < be_va_loss_CE and warmup_weight == 1:
                best_set_CE = True
                be_epoch_num_CE = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if best_set_CE:
                print(
                    'Client %d TRL EMNIST EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d TRL EMNIST EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if loss_CE_va > 1.01 * be_va_loss_CE:
                loss_sat_CE += 1
            else:
                loss_sat_CE = 0

            if loss_sat_CE == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_CE = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')


        self.set_encoder_weights(best_enc_weights)
        self.set_decoder_weights(best_dec_weights)

        # print(best_dec_weights)
        # print(len(best_dec_weights[i]) for i in range(6))
        # print(np.array(best_dec_weights).shape)
        # print(decoder_initial[4:6])
        # sys.exit()

        print('End of EMNIST Classifier Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        self.print_plots('Public EMNIST ', 'Public EMNIST ', self.clientID, tr_overall_acc, va_overall_acc,
                         te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss, te_class_loss,
                         tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                         be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         True, False, True, False)

    def transfer_learning_init_ADV_EMNIST(self):

        print('Starting EMNIST Adversary Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        adv_tr_dict = [self.train_step_ADV_only,
                       self.net_dict['tgt_acc'],
                       self.net_dict['Loss_CE'],
                       self.net_dict['Loss_P1_U_and_yhat_AD'],
                       self.net_dict['Loss_GVIB'],
                       self.net_dict['tgt_predict'],
                       self.net_dict['priv_predict'],
                       self.net_dict['priv_acc']
                       ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_adv = np.inf
        best_set_adv = False
        be_epoch_num_adv = -100

        loss_sat_adv = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1e-3
        learn_rate = 1e-3
        adv_initial = self.get_adv_weights()

        for epoch_num in range(-1, num_transfer_init_adv_epochs):

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.etrain_data, batch_size=100,
                                      num_tr_datapoints=self.enum_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=adv_tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='TRL EM-ADV ')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.eval_data, test_data=self.etest_data, batch_size=100,
                                   num_va_datapoints=self.enum_va_datapoints, num_te_datapoints=self.enum_te_datapoints,
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='TRL EM-ADV ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d TRL EM-ADV EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_adv))
            if epoch_num < 0:
                print('Client %d TRL EM-ADV EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d TRL EM-ADV EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if va_loss_adv < be_va_loss_adv and warmup_weight == 1:
                best_set_adv = True
                be_epoch_num_adv = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_adv_weights = copy.deepcopy(self.get_adv_weights())


            if best_set_adv:
                print(
                    'Client %d TRL EM-ADV EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d TRL EM-ADV EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if va_loss_adv > 1.01 * be_va_loss_adv:
                loss_sat_adv += 1
            else:
                loss_sat_adv = 0

            if loss_sat_adv == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_adv = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')

        # best_adv_weights[4:6] = adv_initial[4:6] # Use copy.deepcopy here ??????????????????????????????????????
        # print('Length of weights: ', len(best_adv_weights))
        self.set_adv_weights(best_adv_weights)


        print('End of EMNIST Adversary Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        print('Client %d Initial Transfer Learning on EMNIST complete' % (self.clientID))
        print('')
        print('')

        self.print_plots('Public EMNIST ', 'Public EMNIST ', self.clientID, tr_overall_acc, va_overall_acc, te_overall_acc,
                         tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss, te_class_loss, tr_adv_loss,
                         va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                         be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         False, True, False, True)

    def transfer_learning_init_MNIST(self):


        print('Starting MNIST Classification Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        tr_dict = [self.train_step_Enc_Dec,
                   self.net_dict['tgt_acc'],
                   self.net_dict['Loss_CE'],
                   self.net_dict['Loss_P1_U_and_yhat_AD'],
                   self.net_dict['Loss_GVIB'],
                   self.net_dict['tgt_predict'],
                   self.net_dict['priv_predict'],
                   self.net_dict['priv_acc']
                   ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_CE = np.inf
        best_set_CE = False
        be_epoch_num_CE = -100

        loss_sat_CE = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1
        learn_rate = 1e-3
        for epoch_num in range(-1, num_transfer_init_enc_dec_epochs):

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.train_data, batch_size=100,
                                      num_tr_datapoints=self.num_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='TRL MNIST ')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_data, test_data=self.test_data, batch_size=100,
                                   num_va_datapoints=self.num_va_datapoints, num_te_datapoints=self.num_te_datapoints,
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='TRL MNIST ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d TRL MNIST EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_CE))
            if epoch_num < 0:
                print('Client %d TRL MNIST EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d TRL MNIST EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if loss_CE_va < be_va_loss_CE and warmup_weight == 1:
                best_set_CE = True
                be_epoch_num_CE = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if best_set_CE:
                print(
                    'Client %d TRL MNIST EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d TRL MNIST EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if loss_CE_va > 1.01 * be_va_loss_CE:
                loss_sat_CE += 1
            else:
                loss_sat_CE = 0

            if loss_sat_CE == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_CE = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')


        self.set_encoder_weights(best_enc_weights)
        self.set_decoder_weights(best_dec_weights)

        print('End of MNIST Classifier Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

    def transfer_learning_init_ADV_MNIST(self):
        # print(self.get_adv_weights())
        # print(self.adversary.get_layer_dict())
        # for layer in self.adversary.layers: print(layer.get_config(), layer.get_weights())
        # self.adversary.layers[0].trainable = False
        # self.adversary.layers[1].trainable = False
        # self.adversary.layers[2].trainable = False
        # self.adversary.layers[3].trainable = False

        print('Starting MNIST Adversary Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        adv_tr_dict = [self.train_step_ADV_only, #self.train_step_ADV_ENC,
                       self.net_dict['tgt_acc'],
                       self.net_dict['Loss_CE'],
                       self.net_dict['Loss_P1_U_and_yhat_AD'],
                       self.net_dict['Loss_GVIB'],
                       self.net_dict['tgt_predict'],
                       self.net_dict['priv_predict'],
                       self.net_dict['priv_acc']
                       ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_adv = np.inf
        best_set_adv = False
        be_epoch_num_adv = -100

        loss_sat_adv = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1e-3
        learn_rate = 1e-3
        for epoch_num in range(-1, num_transfer_init_adv_epochs):

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.train_data, batch_size=100,
                                      num_tr_datapoints=self.num_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=adv_tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='TRL MN-ADV ')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_data, test_data=self.test_data, batch_size=100,
                                   num_va_datapoints=self.num_va_datapoints, num_te_datapoints=self.num_te_datapoints,
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='TRL MN-ADV ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d TRL MN-ADV EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_adv))
            if epoch_num < 0:
                print('Client %d TRL MN-ADV EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d TRL MN-ADV EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if va_loss_adv < be_va_loss_adv and warmup_weight == 1:
                best_set_adv = True
                be_epoch_num_adv = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_adv_weights = copy.deepcopy(self.get_adv_weights())


            if best_set_adv:
                print(
                    'Client %d TRL MN-ADV EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d TRL MN-ADV EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if va_loss_adv > 1.01 * be_va_loss_adv:
                loss_sat_adv += 1
            else:
                loss_sat_adv = 0

            if loss_sat_adv == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_adv = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')

        self.set_adv_weights(best_adv_weights)

        print('End of MNIST Adversary Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        print('Client %d Initial Transfer Learning on MNIST complete' % (self.clientID))
        print('')
        print('')

        self.print_plots('Private MNIST Initial TRFL ', 'Priv-NoGAN MNIST Initial TRFL ', self.clientID, tr_overall_acc,
                         va_overall_acc, te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss,
                         te_class_loss, tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv,
                         be_va_adv_acc, be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         False, True, False, True)

    def transfer_learning_init_EMNIST_MNIST(self):
        print('Starting EMNIST and MNIST Classification Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        tr_dict = [self.train_step_Enc_Dec,
                   self.net_dict['tgt_acc'],
                   self.net_dict['Loss_CE'],
                   self.net_dict['Loss_P1_U_and_yhat_AD'],
                   self.net_dict['Loss_GVIB'],
                   self.net_dict['tgt_predict'],
                   self.net_dict['priv_predict'],
                   self.net_dict['priv_acc']
                   ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_CE = np.inf
        best_set_CE = False
        be_epoch_num_CE = -100

        loss_sat_CE = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1
        learn_rate = 1e-3

        for epoch_num in range(-1, num_transfer_init_enc_dec_epochs):

            self.combine_EMNIST_and_MNIST()

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.train_combined, batch_size=100,
                                      num_tr_datapoints=2*self.num_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='TRL EMNIST and MNIST ')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_combined, test_data=self.test_combined, batch_size=100,
                                   num_va_datapoints=2*self.num_va_datapoints, num_te_datapoints=len(self.test_combined),
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='TRL EMNIST and MNIST ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d TRL EMNIST and MNIST EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_CE))
            if epoch_num < 0:
                print('Client %d TRL EMNIST and MNIST EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d TRL EMNIST and MNIST EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if loss_CE_va < be_va_loss_CE and warmup_weight == 1:
                best_set_CE = True
                be_epoch_num_CE = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if best_set_CE:
                print(
                    'Client %d TRL EMNIST and MNIST EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d TRL EMNIST and MNIST EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if loss_CE_va > 1.01 * be_va_loss_CE:
                loss_sat_CE += 1
            else:
                loss_sat_CE = 0

            if loss_sat_CE == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_CE = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')


        self.set_encoder_weights(best_enc_weights)
        self.set_decoder_weights(best_dec_weights)

        # print(best_dec_weights)
        # print(len(best_dec_weights[i]) for i in range(6))
        # print(np.array(best_dec_weights).shape)
        # print(decoder_initial[4:6])
        # sys.exit()

        print('End of EMNIST and MNIST Classifier Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        parent_dir = "/storage/home/hcoda1/1/asaz3/"
        parent_dir = os.path.join(parent_dir, self.directory)
        sub_dir = "/phase1a_step1_client_" + str(self.clientID)
        # path = os.path.join(parent_dir, sub_dir)
        # try:
        #     os.mkdir(path)
        # except OSError as error:
        #     print(error)
        #     sys.exit()
        # print("Sub-Directory '% s' created" % sub_dir)

        self.print_plots('EMNIST and MNIST ', 'EMNIST and MNIST ', self.clientID, tr_overall_acc, va_overall_acc,
                         te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss, te_class_loss,
                         tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                         be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         True, False, True, False, sub_dir)

    def transfer_learning_init_EMNIST_MNIST_ADV2(self):
        print('Starting EMNIST and MNIST ADV2 Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        tr_dict = [self.train_step_ADV2_only,
                   self.net_dict['adv2_acc'],
                   self.net_dict['Loss_adv2'],
                   self.net_dict['Loss_P1_U_and_yhat_AD'],
                   self.net_dict['Loss_GVIB'],
                   self.net_dict['adv2_predict'],
                   self.net_dict['priv_predict'],
                   self.net_dict['priv_acc']
                   ]

        te_va_dict = [self.net_dict['adv2_acc'],
                      self.net_dict['Loss_adv2'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['adv2_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_CE = np.inf
        best_set_CE = False
        be_epoch_num_CE = -100

        loss_sat_CE = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1
        learn_rate = 1e-3

        for epoch_num in range(-1, num_transfer_init_adv2_epochs):

            self.combine_EMNIST_and_MNIST()

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.train_combined, batch_size=100,
                                      num_tr_datapoints=2*self.num_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='TRL EMNIST and MNIST ADV2')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_combined, test_data=self.test_combined, batch_size=100,
                                   num_va_datapoints=2*self.num_va_datapoints, num_te_datapoints=len(self.test_combined),
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='TRL EMNIST and MNIST ADV2')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d TRL EMNIST and MNIST ADV2 EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta2_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta2_AD,
                                                                                                  loss_sat_CE))
            if epoch_num < 0:
                print('Client %d TRL EMNIST and MNIST ADV2 EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d TRL EMNIST and MNIST ADV2 EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if loss_CE_va < be_va_loss_CE and warmup_weight == 1:
                best_set_CE = True
                be_epoch_num_CE = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if best_set_CE:
                print(
                    'Client %d TRL EMNIST and MNIST ADV2 EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d TRL EMNIST and MNIST ADV2 EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if loss_CE_va > 1.01 * be_va_loss_CE:
                loss_sat_CE += 1
            else:
                loss_sat_CE = 0

            if loss_sat_CE == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_CE = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')


        self.set_encoder_weights(best_enc_weights)
        self.set_decoder_weights(best_dec_weights)

        # print(best_dec_weights)
        # print(len(best_dec_weights[i]) for i in range(6))
        # print(np.array(best_dec_weights).shape)
        # print(decoder_initial[4:6])
        # sys.exit()

        print('End of EMNIST and MNIST ADV2 Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        parent_dir = "/storage/home/hcoda1/1/asaz3/"
        sub_dir = "/phase1a_step1_2_client_" + str(self.clientID)
        path = os.path.join(parent_dir, self.directory + sub_dir)
        # try:
        #     os.mkdir(path)
        # except OSError as error:
        #     print(error)
        #     sys.exit()
        # print("Sub-Directory '% s' created" % sub_dir)

        self.print_plots('EMNIST and MNIST ADV2 ', 'EMNIST and MNIST ADV2 ', self.clientID, tr_overall_acc, va_overall_acc,
                         te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss, te_class_loss,
                         tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                         be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         True, False, True, False, sub_dir)

    def transfer_learning_init_ADV_EMNIST_MNIST(self):

        print('Starting EMNIST and MNIST Adversary Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        adv_tr_dict = [self.train_step_ADV_only,
                       self.net_dict['tgt_acc'],
                       self.net_dict['Loss_CE'],
                       self.net_dict['Loss_P1_U_and_yhat_AD'],
                       self.net_dict['Loss_GVIB'],
                       self.net_dict['tgt_predict'],
                       self.net_dict['priv_predict'],
                       self.net_dict['priv_acc']
                       ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_adv = np.inf
        best_set_adv = False
        be_epoch_num_adv = -100

        loss_sat_adv = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1e-3
        learn_rate = 1e-3
        adv_initial = self.get_adv_weights()

        for epoch_num in range(-1, num_transfer_init_adv_epochs):

            self.combine_EMNIST_and_MNIST()

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.train_combined, batch_size=100,
                                      num_tr_datapoints=2*self.num_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=adv_tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='TRL EM-AND-MN-ADV ')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_combined, test_data=self.test_combined, batch_size=100,
                                   num_va_datapoints=2*self.num_va_datapoints, num_te_datapoints=len(self.test_combined),
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='TRL EM-AND-MN-ADV ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d TRL EM-AND-MN-ADV EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_adv))
            if epoch_num < 0:
                print('Client %d TRL EM-AND-MN-ADV EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d TRL EM-AND-MN-ADV EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if va_loss_adv < be_va_loss_adv and warmup_weight == 1:
                best_set_adv = True
                be_epoch_num_adv = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_adv_weights = copy.deepcopy(self.get_adv_weights())


            if best_set_adv:
                print(
                    'Client %d TRL EM-AND-MN-ADV EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d TRL EM-AND-MN-ADV EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if va_loss_adv > 1.01 * be_va_loss_adv:
                loss_sat_adv += 1
            else:
                loss_sat_adv = 0

            if loss_sat_adv == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_adv = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')

        # best_adv_weights[4:6] = adv_initial[4:6] # Use copy.deepcopy here ??????????????????????????????????????
        # print('Length of weights: ', len(best_adv_weights))
        self.set_adv_weights(best_adv_weights)


        print('End of EMNIST AND MNIST Adversary Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        print('Client %d Initial Transfer Learning complete' % (self.clientID))
        print('')
        print('')

        parent_dir = "/storage/home/hcoda1/1/asaz3/"
        sub_dir = "/phase1a_step2_client_" + str(self.clientID)
        path = os.path.join(parent_dir, self.directory + sub_dir)
        # try:
        #     os.mkdir(path)
        # except OSError as error:
        #     print(error)
        #     sys.exit()
        # print("Sub-Directory '% s' created" % sub_dir)

        self.print_plots('EMNIST AND MNIST ', 'EMNIST AND MNIST ', self.clientID, tr_overall_acc, va_overall_acc, te_overall_acc,
                         tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss, te_class_loss, tr_adv_loss,
                         va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                         be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         False, True, False, True, sub_dir)

    def combine_EMNIST_and_MNIST(self):
        temp_train = np.random.choice(len(self.etrain_data), 500)
        temp_val = np.random.choice(len(self.eval_data), 100)

        self.train_combined = np.concatenate((self.train_data, self.etrain_data[temp_train]), axis=0)
        self.val_combined = np.concatenate((self.val_data, self.eval_data[temp_val]), axis=0)
        self.test_combined = np.concatenate((self.test_data, self.etest_data), axis=0)

        self.train_y_combined = np.ones(2*len(self.train_data))
        self.val_y_combined = np.ones(2*len(self.val_data))
        self.test_y_combined = np.ones(len(self.test_combined))

        self.train_y_combined[0:len(self.train_data)] = np.zeros(len(self.train_data))
        self.val_y_combined[0:len(self.val_data)] = np.zeros(len(self.val_data))
        self.test_y_combined[0:len(self.etest_data)] = np.zeros(len(self.etest_data))

        self.train_combined, self.train_y_combined = shuffle(self.train_combined, self.train_y_combined, random_state=0)
        self.val_combined, self.val_y_combined = shuffle(self.val_combined, self.val_y_combined, random_state=0)
        self.test_combined, self.test_y_combined = shuffle(self.test_combined, self.test_y_combined, random_state=0)

    def compute_all_EMNIST_logits_and_rescale(self):
        data_dict = [self.net_dict['ins_prob']]

        x_batch_tr = self.etrain_data[:, 0:-2]
        priv_batch_tr = self.etrain_data[:, -2]
        tgt_batch_tr = self.etrain_data[:, -1]
        random_tr = np.ones(len(self.etrain_data))
        feed_dict_tr = {self.x: x_batch_tr,
                     self.u: priv_batch_tr,
                     self.v: tgt_batch_tr,
                     self.t: random_tr,
                     self.learning_rate: 1e-3,
                     self.tf_warmup_wt: 1}
        tr_op_dict = self.sess.run(data_dict, feed_dict=feed_dict_tr)
        self.logits_LR_classifier_tr.append(tr_op_dict)
        self.logits_LR_classifier_tr = np.squeeze(np.array(self.logits_LR_classifier_tr))
        self.logits_LR_classifier_tr = np.divide(self.logits_LR_classifier_tr[:, 0], self.logits_LR_classifier_tr[:, 1])


        x_batch_val = self.eval_data[:, 0:-2]
        priv_batch_val = self.eval_data[:, -2]
        tgt_batch_val = self.eval_data[:, -1]
        random_val = np.ones(len(self.eval_data))
        feed_dict_val = {self.x: x_batch_val,
                     self.u: priv_batch_val,
                     self.v: tgt_batch_val,
                     self.t: random_val,
                     self.learning_rate: 1e-3,
                     self.tf_warmup_wt: 1}
        val_op_dict = self.sess.run(data_dict, feed_dict=feed_dict_val)
        self.logits_LR_classifier_val.append(val_op_dict)
        self.logits_LR_classifier_val = np.squeeze(np.array(self.logits_LR_classifier_val))
        self.logits_LR_classifier_val = np.divide(self.logits_LR_classifier_val[:, 0], self.logits_LR_classifier_val[:, 1])

        x_batch_test = self.etest_data[:, 0:-2]
        priv_batch_test = self.etest_data[:, -2]
        tgt_batch_test = self.etest_data[:, -1]
        random_test = np.ones(len(self.etest_data))
        feed_dict_test = {self.x: x_batch_test,
                     self.u: priv_batch_test,
                     self.v: tgt_batch_test,
                     self.t: random_test,
                     self.learning_rate: 1e-3,
                     self.tf_warmup_wt: 1}
        test_op_dict = self.sess.run(data_dict, feed_dict=feed_dict_test)
        self.logits_LR_classifier_test.append(test_op_dict)
        self.logits_LR_classifier_test = np.squeeze(np.array(self.logits_LR_classifier_test))
        self.logits_LR_classifier_test = np.divide(self.logits_LR_classifier_test[:, 0],
                                                  self.logits_LR_classifier_test[:, 1])

        # np.set_printoptions(threshold=np.inf)
        # print(self.logits_LR_classifier_tr)
        # print(self.logits_LR_classifier_val)
        # print(self.logits_LR_classifier_test)

        # print(self.etrain_data[0])

        # BELOW FIXED THE PROBLEM OF CONSTANTLY UPDATED EMNIST DATA!!!!!!!!!!!!!
        self.etrain_data = copy.deepcopy(self.etrain_data_d)
        self.eval_data = copy.deepcopy(self.eval_data_d)
        self.etest_data = copy.deepcopy(self.etest_data_d)


        for ii in range(len(self.etrain_data)):
            self.etrain_data[ii, 0:-2] = self.etrain_data[ii, 0:-2] * self.logits_LR_classifier_tr[ii]
        for jj in range(len(self.eval_data)):
            self.eval_data[jj, 0:-2] = self.eval_data[jj, 0:-2] * self.logits_LR_classifier_val[jj]
        for kk in range(len(self.etest_data)):
            self.etest_data[kk, 0:-2] = self.etest_data[kk, 0:-2] * self.logits_LR_classifier_test[kk]

        # print(self.etrain_data[0])

        #Here is the deal!, look out for the following 3 things:
        #   1) The order of training in def transfer_learning(self)
        #   2) If I need to / can apply my reweighting to loss function directly
        #   3) Feeding both mnist and emnist samples simultaneously

    def local_adversarial_training(self):

        print('Starting Local Adversarial Training over MNIST')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        enc_dec_tr_dict = [self.train_step_GVIB_only,
                           self.net_dict['tgt_acc'],
                           self.net_dict['Loss_CE'],
                           self.net_dict['Loss_P1_U_and_yhat_AD'],
                           self.net_dict['Loss_GVIB'],
                           self.net_dict['tgt_predict'],
                           self.net_dict['priv_predict'],
                           self.net_dict['priv_acc']
                           ]

        adv_tr_dict = [self.train_step_ADV_only,
                       self.net_dict['tgt_acc'],
                       self.net_dict['Loss_CE'],
                       self.net_dict['Loss_P1_U_and_yhat_AD'],
                       self.net_dict['Loss_GVIB'],
                       self.net_dict['tgt_predict'],
                       self.net_dict['priv_predict'],
                       self.net_dict['priv_acc']
                       ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_GVIB = np.inf
        be_va_loss_adversary = np.inf
        best_set_GVIB = False
        be_epoch_num_GVIB = -100

        loss_sat_GVIB = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight_enc_dec = 1e-3
        learn_rate_enc_dec = 1e-4
        warmup_weight_adv = 1e-3
        learn_rate_adv = 1e-4
        for epoch_num in range(-1, num_epochs_adversarial_training_enc_dec):

            if epoch_num >= 0:
                np.random.shuffle(self.train_data)
                train_data_iteration = self.train_data[0:100, :]
                tgt_acc, _, loss_CE, _, py_warmup_weight, tr_time_1 = \
                    self.training_fnc(train_data=train_data_iteration,  # self.train_data,  #
                                      batch_size=100,  # [epoch_num * 100:(epoch_num + 1) * 100, :]
                                      num_tr_datapoints=100,  # self.num_tr_datapoints,
                                      learning_rate_py=learn_rate_enc_dec,
                                      py_warmup_weight=warmup_weight_enc_dec, tr_dict=enc_dec_tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='Local GAN MNIST Classification ')
                tr_overall_acc.append(tgt_acc)
                tr_class_loss.append(loss_CE)
                warmup_weight_enc_dec = py_warmup_weight

                adv_ac = 0
                loss_ad = 0
                tr_timer_2 = 0
                my_bool = False
                for tt in range(10):
                    if tt == 9:
                        my_bool = True

                    _, adv_acc, _, loss_adv, py_warmup_weight, tr_time_2 = \
                        self.training_fnc(train_data=self.train_data, batch_size=100,
                                          num_tr_datapoints=self.num_tr_datapoints, learning_rate_py=learn_rate_adv,
                                          py_warmup_weight=warmup_weight_adv, tr_dict=adv_tr_dict, epoch_num=epoch_num,
                                          print_true=my_bool, string_name='Local GAN MNIST Adversary ')

                    adv_ac = adv_acc
                    loss_ad = loss_adv
                    tr_timer_2 = tr_timer_2 + tr_time_2



                tr_adv_acc.append(adv_ac)
                tr_adv_loss.append(loss_ad)
                warmup_weight_adv = py_warmup_weight

                tr_time = tr_time_1 + tr_timer_2

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_data, test_data=self.test_data, batch_size=100,
                                   num_va_datapoints=self.num_va_datapoints, num_te_datapoints=self.num_te_datapoints,
                                   learning_rate_py=learn_rate_enc_dec, py_warmup_weight=warmup_weight_enc_dec, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='Local GAN Training MNIST ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d Local GAN MNIST EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate_enc_dec,
                                                                                                  warmup_weight_enc_dec,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_GVIB))
            if epoch_num < 0:
                print('Client %d Local GAN MNIST EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d Local GAN MNIST EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if va_gvib_loss < be_va_loss_GVIB:  # and warmup_weight_enc_dec == 1:
                best_set_GVIB = True
                be_epoch_num_GVIB = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if va_loss_adv > be_va_loss_adversary:
                be_va_loss_adversary = copy.deepcopy(va_loss_adv)

            if best_set_GVIB:
                print(
                    'Client %d Local GAN MNIST EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_GVIB,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d Local GAN MNIST EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_GVIB,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            # if va_gvib_loss > 1.01 * be_va_loss_GVIB:
            #     loss_sat_GVIB += 1
            # else:
            #     loss_sat_GVIB = 0
            #
            # if loss_sat_GVIB == loss_sat_lim:
            #     learn_rate_enc_dec *= 0.5
            #     loss_sat_GVIB = 0

            # if va_loss_adv < 1 * be_va_loss_adversary:
            #     loss_sat_GVIB += 1
            # else:
            #     loss_sat_GVIB = 0

            # # Chunk below works well
            # if va_acc_adv < 55:
            #     loss_sat_GVIB += 1
            # else:
            #     loss_sat_GVIB = 0
            #
            # if loss_sat_GVIB == 2:
            #     learn_rate_enc_dec *= 0.5
            #     # learn_rate_adv *= 0.5
            #     loss_sat_GVIB = 0
            # Chunk above works well

            # if learn_rate_enc_dec < 1e-7:
            #     print_logging_info('Converged')
            #     break

            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            if learn_rate_enc_dec < 1e-7: # or va_acc_adv <= 50:
                print('Converged')
                break
            print('')
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING

        # self.set_encoder_weights(best_enc_weights)
        # self.set_decoder_weights(best_dec_weights)
        self.set_encoder_weights(self.get_encoder_weights())
        self.set_decoder_weights(self.get_decoder_weights())

        print('End of Local Adversarial Training over MNIST')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        self.print_plots('Local GAN ', 'Local GAN ', self.clientID, tr_overall_acc,
                         va_overall_acc, te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss,
                         te_class_loss, tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                         be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         True, True, True, True)

    def local_adversarial_training_EMN_and_MN(self):

        print('Starting Local Adversarial Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        enc_dec_tr_dict = [self.train_step_GVIB_only,
                           self.net_dict['tgt_acc'],
                           self.net_dict['Loss_CE'],
                           self.net_dict['Loss_P1_U_and_yhat_AD'],
                           self.net_dict['Loss_GVIB'],
                           self.net_dict['tgt_predict'],
                           self.net_dict['priv_predict'],
                           self.net_dict['priv_acc']
                           ]

        adv_tr_dict = [self.train_step_ADV_only,
                       self.net_dict['tgt_acc'],
                       self.net_dict['Loss_CE'],
                       self.net_dict['Loss_P1_U_and_yhat_AD'],
                       self.net_dict['Loss_GVIB'],
                       self.net_dict['tgt_predict'],
                       self.net_dict['priv_predict'],
                       self.net_dict['priv_acc']
                       ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_GVIB = np.inf
        be_va_loss_adversary = np.inf
        best_set_GVIB = False
        be_epoch_num_GVIB = -100

        loss_sat_GVIB = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight_enc_dec = 1e-3
        learn_rate_enc_dec = 1e-4
        warmup_weight_adv = 1e-3
        learn_rate_adv = 1e-4
        beta_of_adv = 0.05
        for epoch_num in range(-1, num_epochs_adversarial_training_enc_dec):
            self.combine_EMNIST_and_MNIST()

            if epoch_num >= 0:
                np.random.shuffle(self.train_combined)
                train_data_iteration = self.train_combined[0:100, :]
                tgt_acc, _, loss_CE, _, py_warmup_weight, tr_time_1 = \
                    self.training_fnc(train_data=train_data_iteration,  # self.train_data,  #
                                      batch_size=100,  # [epoch_num * 100:(epoch_num + 1) * 100, :]
                                      num_tr_datapoints=100,  # self.num_tr_datapoints,
                                      learning_rate_py=learn_rate_enc_dec,
                                      py_warmup_weight=warmup_weight_enc_dec, tr_dict=enc_dec_tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='Local GAN Classification ', beta=beta_of_adv)
                tr_overall_acc.append(tgt_acc)
                tr_class_loss.append(loss_CE)
                warmup_weight_enc_dec = py_warmup_weight

                adv_ac = 0
                loss_ad = 0
                tr_timer_2 = 0
                my_bool = False
                for tt in range(10):
                    if tt == 9:
                        my_bool = True

                    _, adv_acc, _, loss_adv, py_warmup_weight, tr_time_2 = \
                        self.training_fnc(train_data=self.train_combined, batch_size=100,
                                          num_tr_datapoints=2*self.num_tr_datapoints, learning_rate_py=learn_rate_adv,
                                          py_warmup_weight=warmup_weight_adv, tr_dict=adv_tr_dict, epoch_num=epoch_num,
                                          print_true=my_bool, string_name='Local GAN Adversary ', beta=beta_of_adv)

                    adv_ac = adv_acc
                    loss_ad = loss_adv
                    tr_timer_2 = tr_timer_2 + tr_time_2



                tr_adv_acc.append(adv_ac)
                tr_adv_loss.append(loss_ad)
                warmup_weight_adv = py_warmup_weight

                tr_time = tr_time_1 + tr_timer_2

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_combined, test_data=self.test_combined, batch_size=100,
                                   num_va_datapoints=2*self.num_va_datapoints, num_te_datapoints=len(self.test_combined),
                                   learning_rate_py=learn_rate_enc_dec, py_warmup_weight=warmup_weight_enc_dec, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='Local GAN Training ', beta=beta_of_adv)

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d Local GAN EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate_enc_dec,
                                                                                                  warmup_weight_enc_dec,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_GVIB))
            if epoch_num < 0:
                print('Client %d Local GAN EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d Local GAN EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if va_gvib_loss < be_va_loss_GVIB and warmup_weight_enc_dec == 1:
                best_set_GVIB = True
                be_epoch_num_GVIB = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if va_loss_adv > be_va_loss_adversary:
                be_va_loss_adversary = copy.deepcopy(va_loss_adv)

            if best_set_GVIB:
                print(
                    'Client %d Local GAN EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_GVIB,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d Local GAN EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_GVIB,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            # if va_gvib_loss > 1.01 * be_va_loss_GVIB:
            #     loss_sat_GVIB += 1
            # else:
            #     loss_sat_GVIB = 0
            #
            # if loss_sat_GVIB == loss_sat_lim:
            #     learn_rate_enc_dec *= 0.5
            #     loss_sat_GVIB = 0

            # if va_loss_adv < 1 * be_va_loss_adversary:
            #     loss_sat_GVIB += 1
            # else:
            #     loss_sat_GVIB = 0

            # # Chunk below works well
            # if va_acc_adv < 55:
            #     loss_sat_GVIB += 1
            # else:
            #     loss_sat_GVIB = 0
            #
            # if loss_sat_GVIB == 2:
            #     learn_rate_enc_dec *= 0.5
            #     # learn_rate_adv *= 0.5
            #     loss_sat_GVIB = 0
            # Chunk above works well

            # if learn_rate_enc_dec < 1e-7:
            #     print_logging_info('Converged')
            #     break

            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            if learn_rate_enc_dec < 1e-7: # or va_acc_adv <= 50:
                print('Converged')
                break
            print('')
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING
            # HAVE CHANGED BEST ENC DEC BELOW CORRECT IT BEFORE PROCEEDING

        self.set_encoder_weights(best_enc_weights)
        self.set_decoder_weights(best_dec_weights)
        # self.set_encoder_weights(self.get_encoder_weights())
        # self.set_decoder_weights(self.get_decoder_weights())

        print('End of Local Adversarial Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        parent_dir = "/storage/home/hcoda1/1/asaz3/"
        sub_dir = "/phase1b_client_" + str(self.clientID)
        path = os.path.join(parent_dir, self.directory + sub_dir)
        # try:
        #     os.mkdir(path)
        # except OSError as error:
        #     print(error)
        #     sys.exit()
        # print("Sub-Directory '% s' created" % sub_dir)

        self.print_plots('Local GAN ', 'Local GAN ', self.clientID, tr_overall_acc,
                         va_overall_acc, te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss,
                         te_class_loss, tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                         be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         True, True, True, True, sub_dir)

    def transfer_learning_phase_1a(self):
        self.transfer_learning_init_MNIST()

    def transfer_learning_phase_1b(self):
        print('hi') # self.local_adversarial_training_EMN_and_MN()

    def logits_over_public_data(self):
        logit_dict = [self.net_dict['private_logits_wt_yhat']]
        logit_batch = self.etest_data
        x_batch = logit_batch[:, 0:-2]
        priv_batch = logit_batch[:, -2]
        tgt_batch = logit_batch[:, -1]
        warmup_weight = 1e-3
        learn_rate = 1e-3

        feed_dict = {self.x: x_batch,
                     self.u: priv_batch,
                     self.v: tgt_batch,
                     self.learning_rate: learn_rate,
                     self.tf_warmup_wt: warmup_weight}
        logits = self.sess.run(logit_dict, feed_dict=feed_dict)

        return logits

    def collective_init_ADV_EMNIST(self, col_tr_x, col_tr_y, col_val_x, col_val_y, col_test_x, col_test_y):
        # print(self.get_adv_weights())
        # print(self.adversary.get_layer_dict())
        # for layer in self.adversary.layers: print(layer.get_config(), layer.get_weights())
        # self.adversary.layers[0].trainable = False
        # self.adversary.layers[1].trainable = False
        # self.adversary.layers[2].trainable = False
        # self.adversary.layers[3].trainable = False

        print('Starting Collective Adversary Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        adv_tr_dict = [self.train_step_ADV_reg,
                       self.net_dict['tgt_acc'],
                       self.net_dict['Loss_CE'],
                       self.net_dict['Loss_Collective'],
                       self.net_dict['Loss_GVIB'],
                       self.net_dict['tgt_predict'],
                       self.net_dict['priv_predict'],
                       self.net_dict['priv_acc']
                       ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_Collective'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_adv = np.inf
        best_set_adv = False
        be_epoch_num_adv = -100

        loss_sat_adv = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1e-3
        learn_rate = 1e-3
        for epoch_num in range(-1, num_collective_init_adv_epochs):

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=col_tr_x, batch_size=100,
                                      num_tr_datapoints=self.enum_te_datapoints-2000, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=adv_tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='COL EMN-ADV ', collective_y=col_tr_y)
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=col_val_x, test_data=col_test_x, batch_size=100,
                                   num_va_datapoints=1000, num_te_datapoints=1000,
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='COL EMN-ADV ',
                                   collective_y_vl=col_val_y, collective_y_test=col_test_y)


            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d COL EMN-ADV EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_adv))
            if epoch_num < 0:
                print('Client %d COL EMN-ADV EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d COL EMN-ADV EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if va_loss_adv < be_va_loss_adv and warmup_weight == 1:
                best_set_adv = True
                be_epoch_num_adv = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_adv_weights = copy.deepcopy(self.get_adv_weights())


            if best_set_adv:
                print(
                    'Client %d COL EMN-ADV EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d COL EMN-ADV EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_adv,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if va_loss_adv > 1.01 * be_va_loss_adv:
                loss_sat_adv += 1
            else:
                loss_sat_adv = 0

            if loss_sat_adv == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_adv = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')

        self.set_adv_weights(best_adv_weights)

        print('End of Collective Adversary Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        print('Client %d Collective Transfer Learning on EMNIST complete' % (self.clientID))
        print('')
        print('')

        parent_dir = "/storage/home/hcoda1/1/asaz3/"
        sub_dir = "/phase1a_step3_client_" + str(self.clientID)
        path = os.path.join(parent_dir, self.directory + sub_dir)
        # try:
        #     os.mkdir(path)
        # except OSError as error:
        #     print(error)
        #     sys.exit()
        # print("Sub-Directory '% s' created" % sub_dir)

        self.print_plots('Collective EMNIST TRFL ', 'Coll-NoGAN EMNIST TRFL ', self.clientID, tr_overall_acc,
                         va_overall_acc, te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss,
                         te_class_loss, tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv,
                         be_va_adv_acc, be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                         False, True, False, True, sub_dir)

    def Federated_Learning_MNIST(self, cur_global_epoch_num):


        print('Starting FL Classification Training')
        print('#' * 60)
        print('#' * 60)
        print('#' * 60)

        tr_dict = [self.train_step_GVIB_dec_only,
                   self.net_dict['tgt_acc'],
                   self.net_dict['Loss_CE'],
                   self.net_dict['Loss_P1_U_and_yhat_AD'],
                   self.net_dict['Loss_GVIB'],
                   self.net_dict['tgt_predict'],
                   self.net_dict['priv_predict'],
                   self.net_dict['priv_acc']
                   ]

        te_va_dict = [self.net_dict['tgt_acc'],
                      self.net_dict['Loss_CE'],
                      self.net_dict['Loss_P1_U_and_yhat_AD'],
                      self.net_dict['Loss_GVIB'],
                      self.net_dict['tgt_predict'],
                      self.net_dict['priv_predict'],
                      self.net_dict['priv_acc'],
                      self.net_dict['tgt_logits'],
                      self.net_dict['y_hat']
                      ]

        be_va_loss_CE = np.inf
        best_set_CE = False
        be_epoch_num_CE = -100

        loss_sat_CE = 0

        tr_overall_acc = []
        tr_adv_acc = []
        tr_class_loss = []
        tr_adv_loss = []
        va_overall_acc = []
        va_adv_acc = []
        va_class_loss = []
        va_adv_loss = []
        te_overall_acc = []
        te_adv_acc = []
        te_class_loss = []
        te_adv_loss = []

        warmup_weight = 1e-3
        learn_rate = 1e-3
        for epoch_num in range(-1, num_local_FL_epochs):

            if epoch_num >= 0:
                tgt_acc, adv_acc, loss_CE, loss_adv, py_warmup_weight, tr_time = \
                    self.training_fnc(train_data=self.train_data, batch_size=100,
                                      num_tr_datapoints=self.num_tr_datapoints, learning_rate_py=learn_rate,
                                      py_warmup_weight=warmup_weight, tr_dict=tr_dict, epoch_num=epoch_num,
                                      print_true=True, string_name='FL MNIST ')
                tr_overall_acc.append(tgt_acc)
                tr_adv_acc.append(adv_acc)
                tr_class_loss.append(loss_CE)
                tr_adv_loss.append(loss_adv)
                warmup_weight = py_warmup_weight

            tgt_acc_va, va_acc_adv, loss_CE_va, va_loss_adv, va_gvib_loss, tgt_acc_te, te_acc_adv, loss_CE_te, \
            te_loss_adv, te_gvib_loss, rocauc_va, rocauc_te, va_time, te_time = \
                self.val_test_func(val_data=self.val_data, test_data=self.test_data, batch_size=100,
                                   num_va_datapoints=self.num_va_datapoints, num_te_datapoints=self.num_te_datapoints,
                                   learning_rate_py=learn_rate, py_warmup_weight=warmup_weight, te_va_dict=te_va_dict,
                                   epoch_num=epoch_num, print_val=True, print_test=True, string_name='FL MNIST ')

            va_overall_acc.append(tgt_acc_va)
            va_adv_acc.append(va_acc_adv)
            va_class_loss.append(loss_CE_va)
            va_adv_loss.append(va_loss_adv)
            te_overall_acc.append(tgt_acc_te)
            te_adv_acc.append(te_acc_adv)
            te_class_loss.append(loss_CE_te)
            te_adv_loss.append(te_loss_adv)

            print(
                'Client %d FL MNIST EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                                  epoch_num,
                                                                                                  learn_rate,
                                                                                                  warmup_weight,
                                                                                                  beta1_AD,
                                                                                                  loss_sat_CE))
            if epoch_num < 0:
                print('Client %d FL MNIST EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                 epoch_num,
                                                                                                 va_time,
                                                                                                 te_time))
            else:
                print(
                    'Client %d FL MNIST EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                             epoch_num,
                                                                                             tr_time,
                                                                                             va_time,
                                                                                             te_time))

            if loss_CE_va < be_va_loss_CE and warmup_weight == 1:
                best_set_CE = True
                be_epoch_num_CE = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(tgt_acc_va)
                be_va_loss_CE = copy.deepcopy(loss_CE_va)
                be_va_loss_GVIB = copy.deepcopy(va_gvib_loss)
                be_va_rocauc_score = copy.deepcopy(rocauc_va)
                be_va_adv_acc = copy.deepcopy(va_acc_adv)
                be_va_loss_adv = copy.deepcopy(va_loss_adv)

                be_te_tgt_acc = copy.deepcopy(tgt_acc_te)
                be_te_loss_CE = copy.deepcopy(loss_CE_te)
                be_te_loss_GVIB = copy.deepcopy(te_gvib_loss)
                be_te_rocauc_score = copy.deepcopy(rocauc_te)
                be_te_adv_acc = copy.deepcopy(te_acc_adv)
                be_te_loss_adv = copy.deepcopy(te_loss_adv)

                best_enc_weights = copy.deepcopy(self.get_encoder_weights())
                best_dec_weights = copy.deepcopy(self.get_decoder_weights())

            if best_set_CE:
                print(
                    'Client %d FL MNIST EPOCH %d - BE VA - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_va_tgt_acc,
                    be_va_loss_GVIB,
                    be_va_loss_CE,
                    be_va_loss_adv,
                    be_va_rocauc_score,
                    be_va_adv_acc))

                print(
                    'Client %d FL MNIST EPOCH %d - BE TE - ACC: %3.5g, Loss: %3.5g, CE: %3.5g, ADV_Loss: %3.5g, ROC_AUC: %3.5g, ADV_ACC: %3.5g' % (
                    self.clientID,
                    be_epoch_num_CE,
                    be_te_tgt_acc,
                    be_te_loss_GVIB,
                    be_te_loss_CE,
                    be_te_loss_adv,
                    be_te_rocauc_score,
                    be_te_adv_acc))

            if loss_CE_va > 1.01 * be_va_loss_CE:
                loss_sat_CE += 1
            else:
                loss_sat_CE = 0

            if loss_sat_CE == loss_sat_lim:
                learn_rate *= 0.5
                loss_sat_CE = 0
            if learn_rate < 1e-7:
                print('Converged')
                break
            print('')


        self.set_encoder_weights(best_enc_weights)
        self.set_decoder_weights(best_dec_weights)

        print('End of FL MNIST Classifier Training')
        print('*' * 60)
        print('*' * 60)
        print('*' * 60)
        print(' ' * 60)
        print(' ' * 60)
        print(' ' * 60)

        if cur_global_epoch_num % 10 == 0 or cur_global_epoch_num == 14:
            self.print_plots('FL MNIST ', 'FL-NoGAN MNIST', self.clientID, tr_overall_acc,
                            va_overall_acc, te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc, tr_class_loss, va_class_loss,
                            te_class_loss, tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss_CE, be_va_loss_adv, be_va_adv_acc,
                            be_te_tgt_acc, be_te_loss_CE, be_te_loss_adv, be_te_adv_acc, 100,
                            True, False, True, False)

    def print_plots(self, wh_cl, wh_ad, cl_ID, tr_overall_acc, va_overall_acc, te_overall_acc, tr_adv_acc, va_adv_acc, te_adv_acc,
                    tr_class_loss, va_class_loss, te_class_loss, tr_adv_loss, va_adv_loss, te_adv_loss, be_va_tgt_acc, be_va_loss,
                    be_va_loss_adv, be_va_adv_acc, be_te_tgt_acc, be_te_loss, be_te_loss_adv, be_te_adv_acc, num_clients, p1, p2, p3, p4, sub_dir):
        if p1:
            plt.figure()
            plt.plot(tr_overall_acc, '--ro', label='Training')
            plt.plot(va_overall_acc, '--b*', label='Validation, Best Acc:' + str('%.2f'%be_va_tgt_acc))
            plt.plot(te_overall_acc, '--k^', label='Testing, Best Acc:' + str('%.2f'%be_te_tgt_acc))
            plt.legend()
            plt.title(wh_cl + 'Classifier Accuracy, Client ' + str(cl_ID) + ' of ' + str(num_clients) + ', Beta: ' + str(beta1_AD))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(visible=True)
            plt.savefig(self.directory + '/p' + str(self.clientID) + str(self.plot_counter) + '.png')
            self.plot_counter = self.plot_counter + 1
            plt.show()
        if p2:
            plt.figure()
            plt.plot(tr_adv_acc, '--ro', label='Training')
            plt.plot(va_adv_acc, '--b*', label='Validation, Best Acc:' + str('%.2f'%be_va_adv_acc))
            plt.plot(te_adv_acc, '--k^', label='Testing, Best Acc:' + str('%.2f'%be_te_adv_acc))
            plt.legend()
            plt.title(wh_ad + 'Adversarial Accuracy, Client ' + str(cl_ID) + ' of ' + str(num_clients) + ', Beta: ' + str(beta1_AD))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(visible=True)
            plt.savefig(self.directory + '/p' + str(self.clientID) + str(self.plot_counter) + '.png')
            self.plot_counter = self.plot_counter + 1
            plt.show()
        if p3:
            plt.figure()
            plt.plot(tr_class_loss, '--ro', label='Training')
            plt.plot(va_class_loss, '--b*', label='Validation, Best Loss:' + str('%.2f'%be_va_loss))
            plt.plot(te_class_loss, '--k^', label='Testing, Best Loss:' + str('%.2f'%be_te_loss))
            plt.legend()
            plt.title(wh_cl + 'Classifier Loss, Client ' + str(cl_ID) + ' of ' + str(num_clients) + ', Beta: ' + str(beta1_AD))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(visible=True)
            plt.savefig(self.directory + '/p' + str(self.clientID) + str(self.plot_counter) + '.png')
            self.plot_counter = self.plot_counter + 1
            plt.show()
        if p4:
            plt.figure()
            plt.plot(tr_adv_loss, '--ro', label='Training')
            plt.plot(va_adv_loss, '--b*', label='Validation, Best Loss:' + str('%.2f'%be_va_loss_adv))
            plt.plot(te_adv_loss, '--k^', label='Testing, Best Loss:' + str('%.2f'%be_te_loss_adv))
            plt.legend()
            plt.title(wh_ad + 'Adversarial Loss, Client ' + str(cl_ID) + ' of ' + str(num_clients) + ', Beta: ' + str(beta1_AD))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(visible=True)
            plt.savefig(self.directory + '/p' + str(self.clientID) + str(self.plot_counter) + '.png')
            self.plot_counter = self.plot_counter + 1
            plt.show()

    def post_hoc_adversary(self, tr_all, tr_priv, val_all, val_priv, te_all, te_priv):
        batch_size = self.batch_size
        num_tr_iters = (tr_all.shape)[0] // batch_size
        be_va_loss_CE = np.inf
        tr_dict = [self.train_step_MI_U_and_Yhat_only,
                   self.net_dict['priavte_acc_P1'],
                   self.net_dict['Loss_MI_U_and_Yhat_only']
                   ]

        te_va_dict = [self.net_dict['priavte_acc_P1'],
                      self.net_dict['Loss_MI_U_and_Yhat_only']
                      ]
        learning_rate_py = 1e-3
        py_warmup_weight = 1
        best_set = False
        loss_sat = 0
        be_epoch_num = -100
        for epoch_num in range(-1, num_local_FL_epochs):

            if epoch_num >= 0:
                tr_tgt_acc = 0.0
                tr_loss_CE = 0.0
                tr_time = tm.time()
                for tr_iter in range(num_tr_iters):
                    tr_batch = self.train_data[tr_iter * batch_size:(tr_iter + 1) * batch_size, :]
                    x_batch = tr_batch[:, 0:-2]
                    tgt_batch = tr_batch[:, -1]

                    tr_all_batch = tr_all[tr_iter * batch_size:(tr_iter + 1) * batch_size, :]
                    priv_batch = tr_priv[tr_iter * batch_size:(tr_iter + 1) * batch_size]

                    feed_dict = {self.x: x_batch,
                                 self.u: priv_batch,
                                 self.v: tgt_batch,
                                 self.compressed: tr_all_batch,
                                 self.learning_rate: learning_rate_py,
                                 self.tf_warmup_wt: py_warmup_weight
                                 }
                    tr_op_dict = self.sess.run(tr_dict, feed_dict=feed_dict)
                    tr_tgt_acc += tr_op_dict[1] / float(num_tr_iters)
                    tr_loss_CE += tr_op_dict[2] / float(num_tr_iters)
                tr_time = tm.time() - tr_time
                print(
                    'TESTING Client %d EPOCH %d - TR MI U and Y_hat - ACC: %3.5g, CE: %3.5g' % (
                    self.clientID, epoch_num, tr_tgt_acc, tr_loss_CE))

            va_time = tm.time()
            x_batch = self.val_data[:, 0:-2]
            tgt_batch = self.val_data[:, -1]
            priv_batch = val_priv

            feed_dict = {self.x: x_batch,
                         self.u: priv_batch,
                         self.v: tgt_batch,
                         self.compressed: val_all,
                         self.learning_rate: learning_rate_py,
                         self.tf_warmup_wt: py_warmup_weight}
            va_tgt_acc = 0.0
            va_loss_CE = 0.0
            for klm in range(1):
                va_op_dict = self.sess.run(te_va_dict, feed_dict=feed_dict)
                va_tgt_acc += va_op_dict[0] #/ num_te_va_reps_flt
                va_loss_CE += va_op_dict[1] #/ num_te_va_reps_flt
            print(
                'TESTING Client %d EPOCH %d - VA MI U and Y_hat - ACC: %3.5g, CE: %3.5g' % (
                self.clientID, epoch_num, va_tgt_acc, va_loss_CE))
            va_time = tm.time() - va_time

            te_time = tm.time()
            x_batch = self.test_data[:, 0:-2]
            priv_batch = te_priv
            tgt_batch = self.test_data[:, -1]
            feed_dict = {self.x: x_batch,
                         self.u: priv_batch,
                         self.v: tgt_batch,
                         self.compressed: te_all,
                         self.learning_rate: learning_rate_py,
                         self.tf_warmup_wt: py_warmup_weight}
            te_tgt_acc = 0.0
            te_loss_CE = 0.0
            for _ in range(1):
                te_op_dict = self.sess.run(te_va_dict, feed_dict=feed_dict)
                te_tgt_acc += te_op_dict[0] #/ num_te_va_reps_flt
                te_loss_CE += te_op_dict[1] #/ num_te_va_reps_flt
            te_time = tm.time() - te_time
            print(
                'TESTING Client %d EPOCH %d - TE MI U and Y_hat - ACC: %3.5g, CE: %3.5g' % (
                self.clientID, epoch_num, te_tgt_acc, te_loss_CE))

            if va_loss_CE < be_va_loss_CE:
                best_set = True
                be_epoch_num = copy.deepcopy(epoch_num)

                be_va_tgt_acc = copy.deepcopy(va_tgt_acc)
                be_va_loss_CE = copy.deepcopy(va_loss_CE)

                be_te_tgt_acc = copy.deepcopy(te_tgt_acc)
                be_te_loss_CE = copy.deepcopy(te_loss_CE)

            print(
                'TESTING Client %d EPOCH %d - LR=%3.5g, W_Wt=%3.5g, Beta_AD=%3.5g, Loss_sat=%3.5g' % (self.clientID,
                                                                                              epoch_num,
                                                                                              learning_rate_py,
                                                                                              py_warmup_weight,
                                                                                              beta1_AD,
                                                                                              loss_sat))
            if epoch_num < 0:
                print('TESTING Client %d EPOCH %d - Time taken VA:%3.5gs, TE:%3.5gs' % (self.clientID, epoch_num,
                                                                                             va_time,
                                                                                             te_time))
            else:
                print('TESTING Client %d EPOCH %d - Time taken TR:%3.5gs, VA:%3.5gs, TE:%3.5gs' % (self.clientID,
                                                                                                        epoch_num,
                                                                                                        tr_time,
                                                                                                        va_time,
                                                                                                        te_time))

            if best_set:
                print('TESTING Client %d EPOCH %d - BE VA MI U and Y_hat - ACC: %3.5g, CE: %3.5g' % (
                    self.clientID, be_epoch_num, be_va_tgt_acc, be_va_loss_CE))
                print('TESTING Client %d EPOCH %d - BE TE MI U and Y_hat - ACC: %3.5g, CE: %3.5g' % (
                    self.clientID, be_epoch_num, be_te_tgt_acc, be_te_loss_CE))

        return


    def get_post_hoc_weights(self):
        return self.post_hoc.get_weights()

    def set_post_hoc_weights(self, ph_weights):
        self.post_hoc.set_weights(ph_weights)

    def get_decoder_weights(self):
        return self.decoder.get_weights()

    def set_decoder_weights(self, dec_weights):
        self.decoder.set_weights(dec_weights)

    def get_encoder_weights(self):
        return self.encoder.get_weights()

    def set_encoder_weights(self, enc_weights):
        self.encoder.set_weights(enc_weights)

    def get_adv_weights(self):
        return self.adversary.get_weights()

    def set_adv_weights(self, adv_weights):
        self.adversary.set_weights(adv_weights)

    def get_log_reg_weights(self):
        return self.instance_classifier.get_weights()

    def set_log_reg_weights(self, log_reg_weights):
        self.instance_classifier.set_weights(log_reg_weights)

    def save_compressed_rep(self):

        compressed_dict = [self.net_dict['y_hat']]

        learning_rate_py = 0.1 * 1e-3
        x_batch = self.train_data[:, 0:-2]
        priv_batch = self.train_data[:, -2]
        tgt_batch = self.train_data[:, -1]
        feed_dict = {self.x: x_batch,
                     self.u: priv_batch,
                     self.v: tgt_batch,
                     self.learning_rate: learning_rate_py}
        tr_op_dict = self.sess.run(compressed_dict, feed_dict=feed_dict)
        self.tr_compressed.append(tr_op_dict[0])
        print('Privatized Representations of TRAINING Data of Client %d appended' % self.clientID)

        x_batch = self.val_data[:, 0:-2]
        priv_batch = self.val_data[:, -2]
        tgt_batch = self.val_data[:, -1]
        feed_dict = {self.x: x_batch,
                     self.u: priv_batch,
                     self.v: tgt_batch,
                     self.learning_rate: learning_rate_py}
        va_op_dict = self.sess.run(compressed_dict, feed_dict=feed_dict)
        self.val_compressed.append(va_op_dict[0])

        print('Privatized Representations of VALIDATION Data of Client %d appended' % self.clientID)

        x_batch = self.test_data[:, 0:-2]
        priv_batch = self.test_data[:, -2]
        tgt_batch = self.test_data[:, -1]
        feed_dict = {self.x: x_batch,
                     self.u: priv_batch,
                     self.v: tgt_batch,
                     self.learning_rate: learning_rate_py}

        te_op_dict = self.sess.run(compressed_dict, feed_dict=feed_dict)
        self.te_compressed.append(te_op_dict[0])

        print('Privatized Representations of TESTING Data of Client %d appended' % self.clientID)
        return tr_op_dict[0], va_op_dict[0], te_op_dict[0], self.train_data[:, -1], self.val_data[:, -1], self.test_data[:, -1]

    def get_tr_compressed_rep(self):
        return np.squeeze(np.array(self.tr_compressed)), self.train_data[:, -2], self.train_data[:, -1]

    def get_val_compressed_rep(self):
        return np.squeeze(np.array(self.val_compressed)), self.val_data[:, -2], self.val_data[:, -1]

    def get_te_compressed_rep(self):
        return np.squeeze(np.array(self.te_compressed)), self.test_data[:, -2], self.test_data[:, -1]

def fresh_decoders(clients, random_weights, num_clients):
    for kk in range(num_clients):
        clients[str(kk)].set_decoder_weights(random_weights)

def Fed_Aggregation(clients, num_clients):

    T = 0.1
    batch_size = 100
    noise_multiplier = 1 / (batch_size * np.log(1 / T))
    weight_avg = []


    for mm in range(len(clients[str(0)].get_post_hoc_weights())):
        dividend = np.maximum(1, np.linalg.norm(np.array(clients[str(0)].get_post_hoc_weights())[mm])/4)
        weight_avg.append(np.divide(clients[str(0)].get_post_hoc_weights()[mm], dividend))
        weight_avg[mm] = np.add(weight_avg[mm], np.random.laplace(0, noise_multiplier * 4, weight_avg[mm].size).reshape(weight_avg[mm].shape))
    
    for kk in range(1, num_clients):
        for mm in range(len(clients[str(kk)].get_post_hoc_weights())):
            dividend = np.maximum(1, np.linalg.norm(np.array(clients[str(kk)].get_post_hoc_weights())[mm])/4)
            cur_weight = np.divide(clients[str(kk)].get_post_hoc_weights()[mm], dividend)
            weight_avg[mm] = np.add(weight_avg[mm], cur_weight)
            weight_avg[mm] = np.add(weight_avg[mm], np.random.laplace(0, noise_multiplier * 4, weight_avg[mm].size).reshape(weight_avg[mm].shape))
        
    for jj in range(len(weight_avg)):
        weight_avg[jj] = weight_avg[jj] / num_clients

    for kk in range(num_clients):
        clients[str(kk)].set_post_hoc_weights(weight_avg)

def Fed_Learning(clients, num_clients, tt):
    for kk in range(num_clients):
        clients[str(kk)].Federated_Learning_MNIST(tt)

def Multi_Color_MNIST(MN_x_tr, MN_y_tr, MN_x_vl, MN_y_vl, MN_x_te, MN_y_te):
    MN_x_tr = np.stack([MN_x_tr, MN_x_tr, MN_x_tr], axis=1)
    MN_x_vl = np.stack([MN_x_vl, MN_x_vl, MN_x_vl], axis=1)
    MN_x_te = np.stack([MN_x_te, MN_x_te, MN_x_te], axis=1)
    MN_x_tr = np.squeeze(MN_x_tr)
    MN_x_vl = np.squeeze(MN_x_vl)
    MN_x_te = np.squeeze(MN_x_te)
    MN_x_tr = np.moveaxis(MN_x_tr, 1, -1)
    MN_x_vl = np.moveaxis(MN_x_vl, 1, -1)
    MN_x_te = np.moveaxis(MN_x_te, 1, -1)


    priv_label_tr = np.zeros((len(MN_x_tr), 1))
    priv_label_vl = np.zeros((len(MN_x_vl), 1))
    priv_label_te = np.zeros((len(MN_x_te), 1))

    # set the std of color
    std_color = 0.05

    # set the seed
    np.random.seed(2022)

    color_array = [[230./255., 25./255.,  75./255.],
                   [60./255.,  180./255., 75./255.],
                   [255./255., 225./255., 25./255.],
                   [0./255.,   130./255., 200./255.],
                   [245./255., 130./255., 48./255.],
                   [145./255., 30./255.,  180./255.],
                   [70./255.,  240./255., 240./255.],
                   [240./255., 50./255.,  230./255.],
                   [210./255., 245./255., 60./255.],
                   [250./255., 190./255., 212./255.],
                   [0./255.,   128./255., 128./255.],
                   [220./255., 190./255., 255./255.],
                   [170./255., 110./255., 40./255.],
                   [255./255., 250./255., 200./255.],
                   [128./255., 0./255.,   0./255.],
                   [170./255., 255./255., 195./255.],
                   [128./255., 128./255., 0./255.],
                   [255./255., 215./255., 180./255.],
                   [0./255.,   0./255.,   128./255.],
                   [128./255., 128./255., 128./255.]]

    individual_colors = np.array([np.random.choice(10, 3, replace=False) for _ in range(10)])
    flipping_colors = [i for i in range(3)]
    flipping_colors_vl = [i for i in range(3)]
    flipping_colors_te = [i for i in range(3)]
    flipping_colors = np.array([flipping_colors for _ in range(11)])
    flipping_colors_vl = np.array([flipping_colors_vl for _ in range(11)])
    flipping_colors_te = np.array([flipping_colors_te for _ in range(11)])

    for ii in range(len(MN_x_tr)):
        if int(MN_y_tr[ii]) == 0:
            flipping_colors[0] = np.roll(flipping_colors[0], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[0][flipping_colors[0][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[0][flipping_colors[0][0]]
        elif int(MN_y_tr[ii]) == 1:
            flipping_colors[1] = np.roll(flipping_colors[1], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[1][flipping_colors[1][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[1][flipping_colors[1][0]]
        elif int(MN_y_tr[ii]) == 2:
            flipping_colors[2] = np.roll(flipping_colors[2], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[2][flipping_colors[2][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[2][flipping_colors[2][0]]
        elif int(MN_y_tr[ii]) == 3:
            flipping_colors[3] = np.roll(flipping_colors[3], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[3][flipping_colors[3][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[3][flipping_colors[3][0]]
        elif int(MN_y_tr[ii]) == 4:
            flipping_colors[4] = np.roll(flipping_colors[4], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[4][flipping_colors[4][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[4][flipping_colors[4][0]]
        elif int(MN_y_tr[ii]) == 5:
            flipping_colors[5] = np.roll(flipping_colors[5], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[5][flipping_colors[5][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[5][flipping_colors[5][0]]
        elif int(MN_y_tr[ii]) == 6:
            flipping_colors[6] = np.roll(flipping_colors[6], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[6][flipping_colors[6][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[6][flipping_colors[6][0]]
        elif int(MN_y_tr[ii]) == 7:
            flipping_colors[7] = np.roll(flipping_colors[7], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[7][flipping_colors[7][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[7][flipping_colors[7][0]]
        elif int(MN_y_tr[ii]) == 8:
            flipping_colors[8] = np.roll(flipping_colors[8], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[8][flipping_colors[8][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[8][flipping_colors[8][0]]
        elif int(MN_y_tr[ii]) == 9:
            flipping_colors[9] = np.roll(flipping_colors[9], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[9][flipping_colors[9][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[9][flipping_colors[9][0]]
        elif int(MN_y_tr[ii]) == 10:
            flipping_colors[10] = np.roll(flipping_colors[10], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[0][flipping_colors[10][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_tr[ii] = MN_x_tr[ii] * MN_color
            priv_label_tr[ii] = individual_colors[0][flipping_colors[10][0]]

    for ii in range(len(MN_x_vl)):
        if int(MN_y_vl[ii]) == 0:
            flipping_colors_vl[0] = np.roll(flipping_colors_vl[0], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[0][flipping_colors_vl[0][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[0][flipping_colors_vl[0][0]]
        elif int(MN_y_vl[ii]) == 1:
            flipping_colors_vl[1] = np.roll(flipping_colors_vl[1], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[1][flipping_colors_vl[1][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[1][flipping_colors_vl[1][0]]
        elif int(MN_y_vl[ii]) == 2:
            flipping_colors_vl[2] = np.roll(flipping_colors_vl[2], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[2][flipping_colors_vl[2][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[2][flipping_colors_vl[2][0]]
        elif int(MN_y_vl[ii]) == 3:
            flipping_colors_vl[3] = np.roll(flipping_colors_vl[3], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[3][flipping_colors_vl[3][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[3][flipping_colors_vl[3][0]]
        elif int(MN_y_vl[ii]) == 4:
            flipping_colors_vl[4] = np.roll(flipping_colors_vl[4], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[4][flipping_colors_vl[4][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[4][flipping_colors_vl[4][0]]
        elif int(MN_y_vl[ii]) == 5:
            flipping_colors_vl[5] = np.roll(flipping_colors_vl[5], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[5][flipping_colors_vl[5][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[5][flipping_colors_vl[5][0]]
        elif int(MN_y_vl[ii]) == 6:
            flipping_colors_vl[6] = np.roll(flipping_colors_vl[6], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[6][flipping_colors_vl[6][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[6][flipping_colors_vl[6][0]]
        elif int(MN_y_vl[ii]) == 7:
            flipping_colors_vl[7] = np.roll(flipping_colors_vl[7], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[7][flipping_colors_vl[7][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[7][flipping_colors_vl[7][0]]
        elif int(MN_y_vl[ii]) == 8:
            flipping_colors_vl[8] = np.roll(flipping_colors_vl[8], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[8][flipping_colors_vl[8][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[8][flipping_colors_vl[8][0]]
        elif int(MN_y_vl[ii]) == 9:
            flipping_colors_vl[9] = np.roll(flipping_colors_vl[9], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[9][flipping_colors_vl[9][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[9][flipping_colors_vl[9][0]]
        elif int(MN_y_vl[ii]) == 10:
            flipping_colors_vl[10] = np.roll(flipping_colors_vl[10], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[0][flipping_colors_vl[10][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_vl[ii] = MN_x_vl[ii] * MN_color
            priv_label_vl[ii] = individual_colors[0][flipping_colors_vl[10][0]]

    for ii in range(len(MN_x_te)):
        if int(MN_y_te[ii]) == 0:
            flipping_colors_te[0] = np.roll(flipping_colors_te[0], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[0][flipping_colors_te[0][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[0][flipping_colors_te[0][0]]
        elif int(MN_y_te[ii]) == 1:
            flipping_colors_te[1] = np.roll(flipping_colors_te[1], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[1][flipping_colors_te[1][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[1][flipping_colors_te[1][0]]
        elif int(MN_y_te[ii]) == 2:
            flipping_colors_te[2] = np.roll(flipping_colors_te[2], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[2][flipping_colors_te[2][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[2][flipping_colors_te[2][0]]
        elif int(MN_y_te[ii]) == 3:
            flipping_colors_te[3] = np.roll(flipping_colors_te[3], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[3][flipping_colors_te[3][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[3][flipping_colors_te[3][0]]
        elif int(MN_y_te[ii]) == 4:
            flipping_colors_te[4] = np.roll(flipping_colors_te[4], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[4][flipping_colors_te[4][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[4][flipping_colors_te[4][0]]
        elif int(MN_y_te[ii]) == 5:
            flipping_colors_te[5] = np.roll(flipping_colors_te[5], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[5][flipping_colors_te[5][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[5][flipping_colors_te[5][0]]
        elif int(MN_y_te[ii]) == 6:
            flipping_colors_te[6] = np.roll(flipping_colors_te[6], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[6][flipping_colors_te[6][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[6][flipping_colors_te[6][0]]
        elif int(MN_y_te[ii]) == 7:
            flipping_colors_te[7] = np.roll(flipping_colors_te[7], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[7][flipping_colors_te[7][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[7][flipping_colors_te[7][0]]
        elif int(MN_y_te[ii]) == 8:
            flipping_colors_te[8] = np.roll(flipping_colors_te[8], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[8][flipping_colors_te[8][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[8][flipping_colors_te[8][0]]
        elif int(MN_y_te[ii]) == 9:
            flipping_colors_te[9] = np.roll(flipping_colors_te[9], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[9][flipping_colors_te[9][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[9][flipping_colors_te[9][0]]
        elif int(MN_y_te[ii]) == 10:
            flipping_colors_te[10] = np.roll(flipping_colors_te[10], 1)
            MN_color = np.expand_dims(np.clip(color_array[individual_colors[0][flipping_colors_te[10][0]]] + np.random.randn(1, 3) * std_color, 0.0, 1.0), axis=0)
            MN_x_te[ii] = MN_x_te[ii] * MN_color
            priv_label_te[ii] = individual_colors[0][flipping_colors_te[10][0]]

    return MN_x_tr, MN_x_vl, MN_x_te, priv_label_tr, priv_label_vl, priv_label_te

def load_normalized_Dataset(name, train_val_split):
    # Load training dataset
    ds_MNIST = tfds.load(name)
    ds_train = ds_MNIST['train']
    ds_test = ds_MNIST['test']

    train_np = np.vstack(tfds.as_numpy(ds_train))
    MN_test = np.vstack(tfds.as_numpy(ds_test))

    np.random.seed(2022)
    np.random.shuffle(train_np)
    MN_train, MN_val = np.split(train_np, [train_val_split])

    MN_x_train = np.array(list(map(lambda x: x[0]['image'], MN_train)))
    MN_x_train = np.array(MN_x_train, dtype='f')
    MN_x_train = MN_x_train / 255.
    MN_y_train = np.array(list(map(lambda x: x[0]['label'], MN_train)))
    MN_x_val = np.array(list(map(lambda x: x[0]['image'], MN_val)))
    MN_x_val = np.array(MN_x_val, dtype='f')
    MN_x_val = MN_x_val / 255.
    MN_y_val = np.array(list(map(lambda x: x[0]['label'], MN_val)))
    MN_x_test = np.array(list(map(lambda x: x[0]['image'], MN_test)))
    MN_x_test = np.array(MN_x_test, dtype='f')
    MN_x_test = MN_x_test / 255.
    MN_y_test = np.array(list(map(lambda x: x[0]['label'], MN_test)))

    condition_tr = MN_y_train < 11
    condition_val = MN_y_val < 11
    condition_te = MN_y_test < 11
    MN_y_train = MN_y_train[condition_tr]
    MN_x_train = MN_x_train[condition_tr]
    MN_y_val = MN_y_val[condition_val]
    MN_x_val = MN_x_val[condition_val]
    MN_y_test = MN_y_test[condition_te]
    MN_x_test = MN_x_test[condition_te]

    return MN_x_train, MN_y_train, MN_x_val, MN_y_val, MN_x_test, MN_y_test

def reshape_dataset(num_pixels, MN_x_train, MN_x_val, MN_x_test, MN_y_train, MN_y_val, MN_y_test, priv_label_tr,
                    priv_label_vl, priv_label_te):

    # Reshape E/MNIST to feed it to NN
    MN_x_train = MN_x_train.reshape(MN_x_train.shape[0], 3 * num_pixels)
    MN_x_val = MN_x_val.reshape(MN_x_val.shape[0], 3 * num_pixels)
    MN_x_test = MN_x_test.reshape(MN_x_test.shape[0], 3 * num_pixels)

    # Add dummy adversarial labels as ones array to the position -2!
    MN_train = np.hstack((MN_x_train, priv_label_tr))
    MN_val = np.hstack((MN_x_val, priv_label_vl))
    MN_test = np.hstack((MN_x_test, priv_label_te))

    # Add labels (target values) array to the position -1!
    MN_train = np.hstack((MN_train, np.expand_dims(MN_y_train, axis=1)))
    MN_val = np.hstack((MN_val, np.expand_dims(MN_y_val, axis=1)))
    MN_test = np.hstack((MN_test, np.expand_dims(MN_y_test, axis=1)))

    return MN_train, MN_val, MN_test

def main():

    total_num_clients = 100

    # Start Code here
    # Load normalized MNIST data
    MN_x_train, MN_y_train, MN_x_val, MN_y_val, MN_x_test, MN_y_test = load_normalized_Dataset('mnist', 50000)

    # Load normalized Extended-MNIST data
    EMN_x_train, EMN_y_train, EMN_x_val, EMN_y_val, EMN_x_test, EMN_y_test = load_normalized_Dataset('emnist/letters', 70000)



    # Color MNIST dataset
    MN_x_train, MN_x_val, MN_x_test, priv_label_tr, priv_label_vl, priv_label_te = Multi_Color_MNIST(MN_x_train,
                                                                                                   MN_y_train,
                                                                                                   MN_x_val,
                                                                                                   MN_y_val,
                                                                                                   MN_x_test,
                                                                                                   MN_y_test)

    # Color EMNIST dataset
    EMN_x_train, EMN_x_val, EMN_x_test, epriv_label_tr, epriv_label_vl, epriv_label_te = Multi_Color_MNIST(EMN_x_train,
                                                                                                   EMN_y_train,
                                                                                                   EMN_x_val,
                                                                                                   EMN_y_val,
                                                                                                   EMN_x_test,
                                                                                                   EMN_y_test)
    # # Plot Colored Images
    # for ii in range(30):
    #     plt.imshow(EMN_x_train[ii])
    #     plt.show()
    #     print(epriv_label_tr[ii])


    MN_train, MN_val, MN_test = reshape_dataset(784, MN_x_train, MN_x_val, MN_x_test, MN_y_train, MN_y_val, MN_y_test, priv_label_tr,
                    priv_label_vl, priv_label_te)

    EMN_train, EMN_val, EMN_test = reshape_dataset(784, EMN_x_train, EMN_x_val, EMN_x_test, EMN_y_train-1, EMN_y_val-1, EMN_y_test-1,
                                                epriv_label_tr,
                                                epriv_label_vl, epriv_label_te)

    # Divide dataset into several different users
    local_train_data = np.array_split(MN_train, total_num_clients)
    local_valid_data = np.array_split(MN_val, total_num_clients)


    # Test over ALL of the test data
    local_test_data = []
    for klm in range(total_num_clients):
        local_test_data.append(MN_test)

    num_loc_tr_datapoints = np.shape(local_train_data[0])[0]
    num_loc_va_datapoints = np.shape(local_valid_data[0])[0]
    num_loc_te_datapoints = np.shape(local_test_data[0])[0]

    enum_loc_tr_datapoints = np.shape(EMN_train)[0]
    enum_loc_va_datapoints = np.shape(EMN_val)[0]
    enum_loc_te_datapoints = np.shape(EMN_test)[0]

    # tgt= digit class, priv= colorful or not
    NUM_CLASS_TGT = 10
    NUM_CLASS_PRIVATE = 10

    # Start the session, declare placeholders for main task & adversary
    sess = tf.InteractiveSession()
    x_ip_dim = np.shape(local_train_data)[2] - 2
    x = tf.placeholder(tf.float32, shape=(None, x_ip_dim), name='x_input')  # X - is the input.
    # u = tf.placeholder(tf.float32, shape=(None,), name='u_privacy')  # U - is the privacy label, colorful or not
    u = tf.placeholder_with_default(np.ones(100, dtype=np.float32), shape=(None,), name='u-privacy')  # U - is the privacy label, colorful or not
    v = tf.placeholder(tf.float32, shape=(None,), name='v_privacy')  # V - is the target label, digit class
    t = tf.placeholder(tf.float32, shape=(None,), name='t_class')  # T - is the public/private dataset belonging indicator
    u_TFL = tf.placeholder(tf.float32, shape=(None, 10), name='y_TFL')  # U_TFL - is the label of TFL logits.

    compressed = tf.placeholder_with_default(tf.constant([list(np.zeros(n_latent))]), shape=(None, n_latent), name='compressed')
    learning_rate = tf.placeholder_with_default(1e-4, shape=(), name='learning_rate')
    tf_warmup_wt = tf.placeholder_with_default(1.0, shape=(), name='tf_warmup_wt')
    adv_beta = tf.placeholder_with_default(beta1_AD, shape=(), name='beta_adversary')
    adv_beta2 = tf.placeholder_with_default(beta2_AD, shape=(), name='beta2_adversary')
    indep_beta = tf.placeholder_with_default(indep_AD, shape=(), name='indep_adversary')

    u_int = tf.cast(u, tf.int32)
    v_int = tf.cast(v, tf.int32)
    t_int = tf.cast(t, tf.int32)
    u_one_hot_rep = tf.one_hot(u_int, depth=NUM_CLASS_PRIVATE)
    v_one_hot_rep = tf.one_hot(v_int, depth=NUM_CLASS_TGT)

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    total_num_clients = 4

    #Define all users and relevant models
    for kk in range(total_num_clients):
        tf.set_random_seed(1)
        client_dict[str(kk)] = EdgeNode(local_train_data[kk], local_valid_data[kk], local_test_data[kk], kk, sess,
                                        NUM_CLASS_TGT, NUM_CLASS_PRIVATE, x, u, v, t, u_TFL, u_int, v_int, t_int, compressed,
                                        learning_rate, tf_warmup_wt, adv_beta, num_loc_tr_datapoints, num_loc_va_datapoints,
                                        num_loc_te_datapoints, 100, MN_test, EMN_train, EMN_val, EMN_test,
                                        enum_loc_tr_datapoints, enum_loc_va_datapoints, enum_loc_te_datapoints, "", 784*3,
                                        u_one_hot_rep, adv_beta2, indep_beta)
        client_dict[str(kk)].construct_model()


    # Start training
    sess.run(tf.global_variables_initializer())
    print('Models initialized')

    # Set initial weights before training, all clients have the same
    initial_dec_weights = client_dict[str(0)].get_decoder_weights()
    initial_enc_weights = client_dict[str(0)].get_encoder_weights()
    initial_adv_weights = client_dict[str(0)].get_adv_weights()
    initial_log_reg_weights = client_dict[str(0)].get_log_reg_weights()

    for kk in range(total_num_clients-1):
        client_dict[str(kk+1)].set_decoder_weights(initial_dec_weights)
        client_dict[str(kk+1)].set_encoder_weights(initial_enc_weights)
        client_dict[str(kk+1)].set_adv_weights(initial_adv_weights)
        client_dict[str(kk+1)].set_log_reg_weights(initial_log_reg_weights)

    for kk in range(total_num_clients):
        client_dict[str(kk)].transfer_learning_phase_1a()

    for kk in range(num_global_FL_epochs):
        for jj in range(total_num_clients):
            temp1, temp3, temp5, temp7, temp8, temp9 = client_dict[str(jj)].save_compressed_rep()
            print(temp1.shape)
            print(temp7.shape)
            client_dict[str(jj)].post_hoc_adversary(temp1, temp7, temp3, temp8, temp5, temp9)
        Fed_Aggregation(client_dict, total_num_clients)

    '''client_dict[str(0)].post_hoc_adversary(cr_mn['tr_x'][:], cr_mn['tr_tgt'][:], cr_mn['val_x'][:],
                                            cr_mn['va_tgt'][:], cr_mn['te_x'][:], cr_mn['te_tgt'][:])
    all_compressed_reps_tr = cr_mn['tr_x'][(len(cr_mn['tr_x'])/total_num_clients)*kk : (len(cr_mn['tr_x'])/total_num_clients)*(kk + 1), :]
    all_tgt_tr = cr_mn['tr_tgt'][(len(cr_mn['tr_tgt'])/total_num_clients)*kk : (len(cr_mn['tr_tgt'])/total_num_clients)*(kk + 1), :]

    all_compressed_reps_va = cr_mn['val_x'][(len(cr_mn['val_x'])/total_num_clients)*kk : (len(cr_mn['val_x'])/total_num_clients)*(kk + 1), :]
    all_tgt_va = cr_mn['va_tgt'][(len(cr_mn['va_tgt'])/total_num_clients)*kk : (len(cr_mn['va_tgt'])/total_num_clients)*(kk + 1), :]

    all_compressed_reps_te = cr_mn['te_x'][(len(cr_mn['te_x'])/total_num_clients)*kk : (len(cr_mn['te_x'])/total_num_clients)*(kk + 1), :]
    all_tgt_te = cr_mn['te_tgt'][(len(cr_mn['te_tgt'])/total_num_clients)*kk : (len(cr_mn['te_tgt'])/total_num_clients)*(kk + 1), :]'''

if __name__ == '__main__':
    program_main_st_time = tm.time()
    main()
    # MULTIPROCESSING TRIALS - single main (the 1 line of code above) takes 165 - 170 seconds to execute whereas executing main()
    # via multiprocessing (below) 3 times concurrently (corresponds to 9 models) takes 256.5 to execute
    # p1 = multiprocessing.Process(target=main)
    # p2 = multiprocessing.Process(target=main)
    # p3 = multiprocessing.Process(target=main)
    # p1.start()
    # p2.start()
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()
    current_file_name = os.path.basename(__file__)
    print("Program %s completed in %gs" % (current_file_name, tm.time() - program_main_st_time))
    sys.exit(0)
