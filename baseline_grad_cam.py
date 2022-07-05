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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import *
from model import TorchConvModel
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################



########################################################################
# feature extractor
########################################################################

def list_to_spec_vector_2d_array(file_list,
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
    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_spec_vector_2d_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), frames, n_mels), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :, :] = vector_array

    return dataset


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, 
            file_list,
            param,
            target_source=None,
            ):
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector = list_to_spec_vector_2d_array(self.file_list,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            )
        
    
    def __getitem__(self, index):
        return torch.Tensor(self.data_vector[index, :, :])
    
    def __len__(self):
        return self.data_vector.shape[0]


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
    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))                              
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


########################################################################

class ReconOutputTarget:
    def __init__(self, model_input):
        self.model_input = model_input
        self.error_func = nn.MSELoss(reduction='mean')
    def __call__(self, model_output):
        assert self.model_input.shape == model_output.shape
        error = self.error_func(self.model_input, model_output)
        return error


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
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/valve/id_00".format(base=param["base_directory"]))))

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/bgc_train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_files_pickle = "{pickle}/bgc_eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/ebgc_val_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/bgc_model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/bgc_history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "bgc_{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)

        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_files = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)

            save_pickle(train_pickle, train_files)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        train_dataset = AEDataset(train_files, param)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=param["fit"]["batch_size"], shuffle=True,
        )

        # model training
        print("============== MODEL TRAINING ==============")
        dim_input = train_dataset.data_vector.shape[1]
        model = TorchConvModel().cuda()
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location='cuda:0'))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
            loss_fn = nn.MSELoss()

            for epoch in range(param["fit"]["epochs"]):
                losses = []
                for batch in train_loader:
                    batch = batch.cuda()
                    pred = model(batch)
                    loss = loss_fn(pred, batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                if epoch % 10 == 0:
                    print(f"epoch {epoch}: loss {sum(losses) / len(losses)}")
            model.eval()
            torch.save(model.state_dict(), model_file)

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = [0. for k in eval_labels]
        y_true = eval_labels

        grad_target_layers = [model.ff[6]]
        cam = GradCAM(model=model, target_layers=grad_target_layers, use_cuda=True)

        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            data = file_to_spec_vector_2d_array(file_name,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
            data = torch.Tensor(data).cuda()
            error = torch.mean(((data - model(data)) ** 2), dim=[1, 2])
            y_pred[num] = torch.mean(error).detach().cpu().numpy()

            targets = [ReconOutputTarget(data[0, :, :])]
            cam_result = cam(input_tensor=data, targets=targets)

        score = metrics.roc_auc_score(y_true, y_pred)
        logger.info("anomaly score abnormal : {}".format(str(numpy.array(y_pred)[y_true.astype(bool)])))
        logger.info("anomaly score normal : {}".format(str(numpy.array(y_pred)[numpy.logical_not(y_true)])))
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