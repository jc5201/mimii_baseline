#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""
########################################################################
# import default python-library
########################################################################
import os
import glob
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml
# from import
from tqdm import tqdm
from sklearn import metrics

import torch
from asteroid.models import XUMX

import fast_bss_eval

from utils import *
from model import xumx_model
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################


machine_types = ['fan', 'slider', 'pump', 'valve']


########################################################################
# feature extractor
########################################################################

def train_file_to_mixture_wav(filename):
    machine_type = os.path.split(os.path.split(os.path.split(os.path.split(filename)[0])[0])[0])[1]
    ys = 0
    for machine in machine_types:
        src_filename = filename.replace(machine_type, machine)
        sr, y = file_to_wav_stereo(src_filename)
        ys = ys + y

    return sr, ys
    
def eval_file_to_mixture_wav(filename):
    machine_type = os.path.split(os.path.split(os.path.split(os.path.split(filename)[0])[0])[0])[1]
    ys = 0
    gt_wav = {}
    for normal_type in machine_types:
        if normal_type == machine_type:
            src_filename = filename
        else:
            src_filename = filename.replace(machine_type, normal_type).replace('abnormal', 'normal')
        sr, y = file_to_wav_stereo(src_filename)
        ys = ys + y
        gt_wav[normal_type] = y
    
    return sr, ys, gt_wav

def train_list_to_mixture_waveform_tensor(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    """
    # 01 calculate the number of dimensions
    # dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        sr, ys = train_file_to_mixture_wav(file_list[idx])
        
        vector_array = torch.Tensor(ys)
        # [ch, time]

        if idx == 0:
            dataset = torch.zeros_like(vector_array).unsqueeze(0).repeat(len(file_list), 1, 1)

        dataset[idx, :, :] = vector_array

    return dataset


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """

    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_len = [len(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir.replace('fan', mt),
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext)))) for mt in machine_types]
    normal_len = min(min(normal_len), len(normal_files))
    normal_files = normal_files[:normal_len]

    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir.replace('fan', '*'),
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    num_eval_normal = 250
    train_files = normal_files[num_eval_normal:]
    train_labels = normal_labels[num_eval_normal:]
    eval_files = numpy.concatenate((normal_files[:num_eval_normal], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:num_eval_normal], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels



########################################################################
# main
########################################################################
if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/fan/id_04".format(base=param["base_directory"]))))  # {base}/0dB/fan/id_00/normal/00000000.wav
    # dirs = sorted(glob.glob(os.path.abspath("{base}/*/fan/*".format(base=param["base_directory"]))))  # {base}/0dB/fan/id_00/normal/00000000.wav

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = 'mix'
        machine_id = os.path.split(target_dir)[1]
        # target_dir = target_dir.replace('fan', '*')

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}_waveform.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)

        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        if os.path.exists(train_pickle):
            if os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
                eval_files = load_pickle(eval_files_pickle)
                eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)

            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        # TODO
        # model load
        print("============== MODEL LOADING ==============")
        if db == '0dB':
            if machine_id == 'id_00':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_0dB_id0/checkpoints/epoch=41-step=1049.ckpt'
            elif machine_id == 'id_02':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_0dB_id2/checkpoints/epoch=33-step=849.ckpt'
            elif machine_id == 'id_04':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_0dB_id4/checkpoints/epoch=40-step=1024.ckpt'
            elif machine_id == 'id_06':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_0dB_id6/checkpoints/epoch=37-step=949.ckpt'
            else:
                model_path = '/hdd/hdd1/sss/xumx/0613_vanilla_0dB/checkpoints/epoch=26-step=2024.ckpt'
        elif db == '6dB':
            if machine_id == 'id_00':
                model_path = '/hdd/hdd1/sss/xumx/0617_5_vanilla_6dB_id0/checkpoints/epoch=21-step=1363.ckpt'
                # model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_6dB_id0/checkpoints/epoch=32-step=824.ckpt'
            elif machine_id == 'id_02':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_6dB_id2/checkpoints/epoch=33-step=849.ckpt'
            elif machine_id == 'id_04':
                model_path = '/hdd/hdd1/sss/xumx/0617_5_vanilla_no_mute_6dB_id4/checkpoints/epoch=39-step=1359.ckpt'
                # model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_6dB_id4/checkpoints/epoch=41-step=1049.ckpt'
            elif machine_id == 'id_06':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_6dB_id6/checkpoints/epoch=35-step=899.ckpt'
            else:
                model_path = '/hdd/hdd1/sss/xumx/0613_vanilla_6dB/checkpoints/epoch=16-step=1274.ckpt'
        elif db == 'min6dB':
            if machine_id == 'id_00':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_min6dB_id0/checkpoints/epoch=66-step=1674.ckpt'
            elif machine_id == 'id_02':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_min6dB_id2/checkpoints/epoch=63-step=1599.ckpt'
            elif machine_id == 'id_04':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_min6dB_id4/checkpoints/epoch=909-step=22749.ckpt'
            elif machine_id == 'id_06':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_min6dB_id6/checkpoints/epoch=60-step=1524.ckpt'
            else:
                model_path = '/hdd/hdd1/sss/xumx/0613_vanilla_min6dB/checkpoints/epoch=19-step=1499.ckpt'
        else:
            raise Exception('dB not found')
        model = xumx_model(model_path)
        model.eval()
        model = model.cuda()

        logger.info(f"loading model <- {model_path}")

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = numpy.array([0. for k in eval_labels])
        y_true = numpy.array(eval_labels)
        
        y_pred_types = {mt: numpy.copy(y_pred) for mt in machine_types}
        y_true_types = {mt: numpy.copy(y_true) for mt in machine_types}

        eval_types = {mt: [] for mt in machine_types}
        # ys = 0
        # for machine in machine_types:
        #     filename = file_list[idx].replace('fan', machine)
        #     sr, y = file_to_wav(filename)
        #     ys = ys + y
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            # try:
            machine_type = os.path.split(os.path.split(os.path.split(os.path.split(file_name)[0])[0])[0])[1]

            sr, ys, gt_wav = eval_file_to_mixture_wav(file_name)

            data = torch.Tensor(ys).cuda()

            #
            t_data = data.unsqueeze(0)
            # [B, Ch, time]
            pred_spec, pred_wav, mix = model(t_data, return_mixture=True)
            # pred_spec [src, Tb, B, ch, fb]
            # pred_wav [src, B, ch, time]

            # import soundfile
            # soundfile.write("ch0.wav", pred_wav[0, 0].permute(1, 0).detach().cpu().numpy(), 16000)
            # soundfile.write("ch1.wav", pred_wav[1, 0].permute(1, 0).detach().cpu().numpy(), 16000)
            # soundfile.write("ch2.wav", pred_wav[2, 0].permute(1, 0).detach().cpu().numpy(), 16000)
            # soundfile.write("ch3.wav", pred_wav[3, 0].permute(1, 0).detach().cpu().numpy(), 16000)

            ## spec
            # recon_error = torch.sum(pred_spec, dim=0) - mix
            # error = torch.mean(torch.norm(recon_error, dim=[0, 3]))

            ## wav
            # recon_error = torch.sum(pred_wav, dim=0) - t_data[:, :, :pred_wav.shape[3]]
            # error = torch.mean(torch.norm(recon_error, dim=2))
            
            ## wav_src
            ref = torch.stack([torch.Tensor(gt_wav[machine_type]).unsqueeze(0) for machine_type in machine_types], dim=0)
            recon_sisdr = fast_bss_eval.sdr(ref[:, :, :, :pred_wav.shape[3]].cuda(), pred_wav)
            machine_idx = machine_types.index(machine_type)
            error = -torch.mean(recon_sisdr[machine_idx, :, :])
            
            y_pred[num] = error.detach().cpu().numpy()
            if num <= 250:
                for i, mt in enumerate(machine_types):
                    eval_types[mt].append(num)
                    error = -torch.mean(recon_sisdr[i, :, :]).detach().cpu().numpy()
                    y_pred_types[mt][num] = error
                    y_true_types[mt][num] = y_true[num]
            else:
                eval_types[machine_type].append(num)
                y_pred_types[machine_type][num] = error.detach().cpu().numpy()
                y_true_types[machine_type][num] = y_true[num]

        scores = []
        for machine_type in machine_types:
            y_pred_mt = y_pred_types[machine_type][eval_types[machine_type]]
            y_pred_mt_ = numpy.exp(y_pred_mt)/sum(numpy.exp(y_pred_mt))
            score = metrics.roc_auc_score(y_true_types[machine_type][eval_types[machine_type]], y_pred_mt_)
            logger.info("AUC_{} : {}".format(machine_type, score))
            evaluation_result["AUC_{}".format(machine_type)] = float(score)
            scores.append(score)
        score = sum(scores) / len(scores)
        logger.info("AUC : {}".format(score))
        evaluation_result["AUC"] = float(score)
        results[evaluation_result_key] = evaluation_result
        print("===========================")

    # output results
    print("\n===========================")
    logger.info("all results -> {}".format(result_file))
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")
########################################################################
