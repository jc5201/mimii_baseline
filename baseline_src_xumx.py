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
import sys
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
import torch.nn as nn
import fast_bss_eval
import museval

from utils import *
from model import TorchModel, xumx_model
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################

machine_types = ['id_00', 'id_02']
num_eval_normal = 250


########################################################################
# feature extractor
########################################################################

def generate_label(y):
    rms_fig = librosa.feature.rms(y=y)
    rms_tensor = torch.tensor(rms_fig).reshape(1, -1, 1)
    rms_trim = rms_tensor.expand(-1, -1, 512).reshape(1, -1)[:, :160000]

    k = int(y.shape[1]*0.8)
    min_threshold, _ = torch.kthvalue(rms_trim, k)
    label = (rms_trim > min_threshold).type(torch.float)
    label = label.expand(y.shape[0], -1)
    return label


def train_file_to_mixture_wav_label(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    active_label_sources = {}
    for machine in machine_types:
        src_filename = filename.replace(machine_type, machine)
        sr, y = file_to_wav_stereo(src_filename)
        active_label_sources[machine] = generate_label(y)
        ys = ys + y

    return sr, ys, active_label_sources


def eval_file_to_mixture_wav_label(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    gt_wav = {}
    active_label_sources = {}
    for normal_type in machine_types:
        if normal_type == machine_type:
            src_filename = filename
        else:
            src_filename = filename.replace(machine_type, normal_type).replace('abnormal', 'normal')
        sr, y = file_to_wav_stereo(src_filename)
        ys = ys + y
        active_label_sources[normal_type] = generate_label(y)
        gt_wav[normal_type] = y
    
    return sr, ys, gt_wav, active_label_sources


def train_list_to_mix_sep_spec_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         sep_model=None,
                         target_source=None):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):
        target_type = os.path.split(os.path.split(os.path.split(file_list[idx])[0])[0])[1]

        sr, mixture_y, active_label_sources = train_file_to_mixture_wav_label(file_list[idx])
            
        active_labels = torch.stack([active_label_sources[src] for src in machine_types])
        _, time = sep_model(torch.Tensor(mixture_y).unsqueeze(0).cuda(), active_labels.unsqueeze(0).cuda())
        # [src, b, ch, time]
        if target_source is not None:
            target_idx = machine_types.index(target_source)
        else:
            target_idx = machine_types.index(target_type)
        ys = time[target_idx, 0, 0, :].detach().cpu().numpy()
        
        vector_array = wav_to_spec_vector_array(sr, ys,
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset

class AEDataset(torch.utils.data.Dataset):
    def __init__(self, 
            sep_model, 
            file_list,
            param,
            target_source=None,
            ):
        self.sep_model = sep_model
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector = train_list_to_mix_sep_spec_vector_array(self.file_list,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            sep_model=sep_model,
                                            target_source=target_source)
        
    
    def __getitem__(self, index):
        return torch.Tensor(self.data_vector[index, :])
    
    def __len__(self):
        return self.data_vector.shape[0]


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):

    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                ext=ext))))
    normal_len = [len(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir.replace('id_00', mt),
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext)))) for mt in machine_types]   #dataset 중에서 가장 짧은 것 기준
    normal_len = min(min(normal_len), len(normal_files))
    normal_files = normal_files[:normal_len]


    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = []
    for machine_type in machine_types:
        abnormal_files.extend(sorted(glob.glob(
            os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir.replace('id_00', machine_type),
                                                                 abnormal_dir_name=abnormal_dir_name,
                                                                 ext=ext)))))   
    abnormal_files.extend(sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir.replace('id_00', 'id_02'),
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                 ext=ext)))))                                               
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[num_eval_normal:]
    train_labels = normal_labels[num_eval_normal:]
    eval_normal_files = sum([[fan_file.replace("id_00", machine_type) for fan_file in normal_files[:num_eval_normal]] for machine_type in machine_types], [])
    eval_files = numpy.concatenate((eval_normal_files, abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:num_eval_normal], normal_labels[:num_eval_normal], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


########################################################################



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


    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/valve/id_00".format(base=param["base_directory"]))))  # {base}/0dB/fan/id_00/normal/00000000.wav
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
        machine_id = os.path.split(target_dir)[1]  ##TODO: machine id 고치기
        # target_dir = target_dir.replace('fan', '*')

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/src_train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_files_pickle = "{pickle}/src_eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/src_eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/src_model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/src_history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)
   

        model_path = '/hdd/hdd1/lyj/xumx/w-control/epoch=985-step=44369.ckpt'

        sep_model = xumx_model(model_path)
        sep_model.eval()
        sep_model = sep_model.cuda()


        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        # if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
        #     train_files, train_labels = load_pickle(train_pickle)
        #     eval_files = load_pickle(eval_files_pickle)
        #     eval_labels = load_pickle(eval_labels_pickle)
        # else:
        train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
        save_pickle(train_pickle, (train_files, train_labels))
        save_pickle(eval_files_pickle, eval_files)
        save_pickle(eval_labels_pickle, eval_labels)
        
        model = {}
        for target_type in machine_types:

            train_dataset = AEDataset(sep_model, train_files, param, target_source=target_type)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=param["fit"]["batch_size"], shuffle=True,
            )

            # model training
            print("============== MODEL TRAINING ==============")
            dim_input = train_dataset.data_vector.shape[1]
            model[target_type] = TorchModel(dim_input).cuda()
            optimizer = torch.optim.Adam(model[target_type].parameters(), lr=1.0e-4)
            loss_fn = nn.MSELoss()

            for epoch in range(param["fit"]["epochs"]):
                losses = []
                for batch in train_loader:
                    batch = batch.cuda()
                    pred = model[target_type](batch)
                    loss = loss_fn(pred, batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                if epoch % 10 == 0:
                    print(f"epoch {epoch}: loss {sum(losses) / len(losses)}")
            model[target_type].eval()
               
        # evaluation
        print("============== EVALUATION ==============")
        y_pred = numpy.array([0. for k in eval_labels])
        y_true = numpy.array(eval_labels)
        sdr_pred_normal = {mt: [] for mt in machine_types}
        sdr_pred_abnormal = {mt: [] for mt in machine_types}

        eval_types = {mt: [] for mt in machine_types}
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            machine_type = os.path.split(os.path.split(os.path.split(file_name)[0])[0])[1]
            target_idx = machine_types.index(machine_type)

            sr, mixture_y, y_raw, active_label_sources = eval_file_to_mixture_wav_label(file_name)

            active_labels = torch.stack([active_label_sources[src] for src in machine_types])
            _, time = sep_model(torch.Tensor(mixture_y).unsqueeze(0).cuda(), active_labels.unsqueeze(0).cuda())
            # [src, b, ch, time]
            ys = time[target_idx, 0, 0, :].detach().cpu().numpy()
            
            data = wav_to_spec_vector_array(sr, ys,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
            data = torch.Tensor(data).cuda()
            error = torch.mean(((data - model[machine_type](data)) ** 2), dim=1)

            sep_sdr, _, _, _ = museval.evaluate(numpy.expand_dims(y_raw[machine_type][0, :ys.shape[0]], axis=(0,2)), 
                                        numpy.expand_dims(ys, axis=(0,2)))

            y_pred[num] = torch.mean(error).detach().cpu().numpy()
            eval_types[machine_type].append(num)

            if num < num_eval_normal * 2: # normal file
                sdr_pred_normal[machine_type].append(numpy.mean(sep_sdr))
            else: # abnormal file
                sdr_pred_abnormal[machine_type].append(numpy.mean(sep_sdr))

        scores = []
        for machine_type in machine_types:
            score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred[eval_types[machine_type]])

            # code to dump errors
            # logger.info("anomaly score abnormal : {}".format(str(numpy.array(y_pred[eval_types[machine_type]])[y_true[eval_types[machine_type]].astype(bool)])))
            # logger.info("anomaly score normal : {}".format(str(numpy.array(y_pred[eval_types[machine_type]])[numpy.logical_not(y_true[eval_types[machine_type]])])))

            logger.info("AUC_{} : {}".format(machine_type, score))
            evaluation_result["AUC_{}".format(machine_type)] = float(score)
            scores.append(score)
            logger.info("SDR_normal_{} : {}".format(machine_type, sum(sdr_pred_normal[machine_type])/len(sdr_pred_normal[machine_type])))
            logger.info("SDR_abnormal_{} : {}".format(machine_type, sum(sdr_pred_abnormal[machine_type])/len(sdr_pred_abnormal[machine_type])))
            evaluation_result["SDR_normal_{}".format(machine_type)] = float(sum(sdr_pred_normal[machine_type])/len(sdr_pred_normal[machine_type]))
            evaluation_result["SDR_abnormal_{}".format(machine_type)] = float(sum(sdr_pred_abnormal[machine_type])/len(sdr_pred_abnormal[machine_type]))
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