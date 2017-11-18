#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    标记文件
"""
import codecs
import yaml
import pickle
import tensorflow as tf
from load_data import load_vocs, init_data
from model import SequenceLabelingModel
from collections import defaultdict as mydict


def main():
    # 加载配置文件
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    feature_names = config['model_params']['feature_names']

    # 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化)
    feature_weight_shape_dict, feature_weight_dropout_dict, \
        feature_init_weight_dict = dict(), dict(), dict()
    for feature_name in feature_names:
        feature_weight_shape_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['shape']
        feature_weight_dropout_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['dropout_rate']
        path_pre_train = config['model_params']['embed_params'][feature_name]['path']
        if path_pre_train:
            with open(path_pre_train, 'rb') as file_r:
                feature_init_weight_dict[feature_name] = pickle.load(file_r)

    # 加载数据

    # 加载vocs
    path_vocs = []
    for feature_name in feature_names:
        path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
    path_vocs.append(config['data_params']['voc_params']['label']['path'])
    vocs = load_vocs(path_vocs)

    # 加载数据
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']
    sep = '\t' if sep_str == 'table' else ' '
    data_dict = init_data(
        path=config['data_params']['path_test'], feature_names=feature_names, sep=sep,
        vocs=vocs, max_len=config['model_params']['sequence_length'], model='test')

    # 加载模型
    model = SequenceLabelingModel(
        sequence_length=config['model_params']['sequence_length'],
        nb_classes=config['model_params']['nb_classes'],
        nb_hidden=config['model_params']['bilstm_params']['num_units'],
        feature_weight_shape_dict=feature_weight_shape_dict,
        feature_init_weight_dict=feature_init_weight_dict,
        feature_weight_dropout_dict=feature_weight_dropout_dict,
        dropout_rate=config['model_params']['dropout_rate'],
        nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
        batch_size=config['model_params']['batch_size'],
        train_max_patience=config['model_params']['max_patience'],
        use_crf=config['model_params']['use_crf'],
        l2_rate=config['model_params']['l2_rate'],
        rnn_unit=config['model_params']['rnn_unit'],
        learning_rate=config['model_params']['learning_rate'],
        path_model=config['model_params']['path_model'])
    saver = tf.train.Saver()
    saver.restore(model.sess, config['model_params']['path_model'])

    # 标记
    viterbi_sequences = model.predict(data_dict)

    # 写入文件
    label_voc = dict()
    for key in vocs[-1]:
        label_voc[vocs[-1][key]] = key
    with codecs.open(config['data_params']['path_test'], 'r', encoding='utf-8') as file_r:
        sentences = file_r.read().strip()
        sentences = sentences.replace('\r','')
        sentences = sentences.split('\n\n\n')
    file_result = codecs.open(
        config['data_params']['path_result'], 'w', encoding='utf-8')

    Entityset = mydict(set)
    EntityDetectset = mydict(set)
    EntityCorrectset = mydict(set)

    flag_0 = False
    flag_1 = False
    flag_2 = False

    entity_name = ''
    entity_detect_name = ''
    entity_correct_name = ''
    for i, sentence in enumerate(sentences):
        for j, item in enumerate(sentence.split('\n')):
            true_label = item.split('\t')[-1].rstrip()
            word = item.split('\t')[0].rstrip()
            if j < len(viterbi_sequences[i]):
                mark_label = label_voc[viterbi_sequences[i][j]]
            else:
                mark_label = 'O'
            #统计真实的情况
            if('B_' in true_label):
                entity_name = word
                flag_0 = True
            elif('I_' in true_label and flag_0):
                entity_name += word
            elif('E_' in true_label and flag_0):
                entity_name += word
                Entityset[true_label].add(entity_name)
                Entityset['Total'].add(entity_name)
                entity_name = ''
                flag_0 = False
            else:
                entity_name = ''
                flag_0 = False

            #统计检测到的情况
            if('B_' in mark_label):
                entity_detect_name = word
                flag_1 = True
            elif('I_' in mark_label and flag_1):
                entity_detect_name += word
            elif('E_' in mark_label and flag_1):
                entity_detect_name += word
                EntityDetectset[mark_label].add(entity_detect_name)
                EntityDetectset['Total'].add(entity_detect_name)
                entity_detect_name = ''
                flag_1 = False
            else:
                entity_detect_name = ''
                flag_1 = False

            #统计正确的情况
            if(true_label == mark_label):
                if('B_' in true_label):
                    flag_2 = True
                    entity_correct_name = word
                elif('I_' in true_label and flag_2):
                    entity_correct_name += word
                elif('E_' in true_label and flag_2):
                    entity_correct_name += word
                    EntityCorrectset[true_label].add(entity_correct_name)
                    EntityCorrectset['Total'].add(entity_correct_name)
                    entity_correct_name = ''
                else:
                    entity_correct_name = ''
                    flag_2 = False
            else:
                entity_correct_name = ''
                flag_2 = False

            if j < len(viterbi_sequences[i]):
                file_result.write('%s\t%s\r\n' % (item, mark_label))
            else:
                file_result.write('%s\tO\r\n' % item)
        file_result.write('\r\n')

    file_result.close()

    printSetState(Entityset,"True")
    printSetState(EntityCorrectset,"Correct")
    printSetState(EntityDetectset,"Detect")


def printSetState(entity_set, name):
    print("The state of set %s: " % name)
    for item in entity_set:
        print('item %s have %d elements' % (item,len(entity_set[item])))




if __name__ == '__main__':
    main()
