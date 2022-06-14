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
import pickle
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
import logging
# from import
from tqdm import tqdm
from sklearn import metrics

import torch
from asteroid.models import XUMX

import fast_bss_eval
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save .png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


########################################################################
# file I/O
########################################################################
# pickle I/O
def save_pickle(filename, save_data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data


# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=1):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, numpy.array(multi_channel_data)[:channel, :]

    except ValueError as msg:
        logger.warning(f'{msg}')


########################################################################


########################################################################
# feature extractor
########################################################################

def file_to_wav(file_name):
    sr, y = demux_wav(file_name, channel=2)
    return sr, y


def train_list_to_vector_array(file_list,
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
    # 01 calculate the number of dimensions
    # dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        machine_types = ['fan', 'slider', 'pump', 'valve']
        ys = 0
        for machine in machine_types:
            filename = file_list[idx].replace('fan', machine)
            sr, y = file_to_wav(filename)
            ys = ys + y
        
        vector_array = torch.Tensor(ys)
        # [ch, time]

        # vector_array = wav_to_vector_array(sr, ys,
        #                                     n_mels=n_mels,
        #                                     frames=frames,
        #                                     n_fft=n_fft,
        #                                     hop_length=hop_length,
        #                                     power=power)

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
    machine_types = ['fan', 'slider', 'pump', 'valve']
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


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = numpy.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)

    return numpy.max(numpy.where(freqs <= bandwidth)[0]) + 1

class XUMXSystem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None


def xumx_model(path):
    
    x_unmix = XUMX(
        window_length=4096,
        input_mean=None,
        input_scale=None,
        nb_channels=2,
        hidden_size=512,
        in_chan=4096,
        n_hop=1024,
        sources=["fan", "pump", "slider", "valve"],
        max_bin=bandwidth_to_max_bin(16000, 4096, 16000),
        bidirectional=True,
        sample_rate=16000,
        spec_power=1,
        return_time_signals=True,
    )

    conf = torch.load(path, map_location="cpu")

    system = XUMXSystem()
    system.model = x_unmix

    system.load_state_dict(conf['state_dict'], strict=False)

    return system.model



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

    machine_types = ['fan', 'slider', 'pump', 'valve']

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/*/fan/*".format(base=param["base_directory"]))))  # {base}/0dB/fan/id_00/normal/00000000.wav
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
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_6dB_id0/checkpoints/epoch=32-step=824.ckpt'
            elif machine_id == 'id_02':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_6dB_id2/checkpoints/epoch=33-step=849.ckpt'
            elif machine_id == 'id_04':
                model_path = '/hdd/hdd1/sss/xumx/0614_vanilla_6dB_id4/checkpoints/epoch=41-step=1049.ckpt'
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
        # model = xumx_model('/hdd/hdd1/sss/xumx/0613_vanilla_0dB/checkpoints/???')
        model.eval()
        model = model.cuda()

        logger.info(f"loading model <- {model_path}")

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = numpy.array([0. for k in eval_labels])
        y_true = numpy.array(eval_labels)
        
        y_pred_types = {mt: numpy.copy(y_pred) for mt in machine_types}
        y_true_types = {mt: numpy.copy(y_true) for mt in machine_types}

        machine_types = ['fan', 'slider', 'pump', 'valve']
        eval_types = {mt: [] for mt in machine_types}
        # ys = 0
        # for machine in machine_types:
        #     filename = file_list[idx].replace('fan', machine)
        #     sr, y = file_to_wav(filename)
        #     ys = ys + y
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            try:
                machine_type = os.path.split(os.path.split(os.path.split(os.path.split(file_name)[0])[0])[0])[1]
                ys = 0
                gt_wav = {}
                for normal_type in machine_types:
                    if normal_type == machine_type:
                        continue
                    normal_file_name = file_name.replace(machine_type, normal_type).replace('abnormal', 'normal')
                    sr, y = demux_wav(normal_file_name, channel=2)
                    ys += y
                    gt_wav[normal_type] = y
                    
                sr, y = demux_wav(file_name, channel=2)
                ys += y
                gt_wav[machine_type] = y

                data = torch.Tensor(ys).cuda()
                # 

                #
                t_data = data.unsqueeze(0)
                # [B, Ch, time]
                pred_spec, pred_wav, mix = model(t_data, return_mixture=True)
                # pred_spec [src, Tb, B, ch, fb]
                # pred_wav [src, B, ch, time]

                ## spec
                # recon_error = torch.sum(pred_spec, dim=0) - mix
                # error = torch.mean(torch.norm(recon_error, dim=[0, 3]))

                ## wav
                # recon_error = torch.sum(pred_wav, dim=0) - t_data[:, :, :pred_wav.shape[3]]
                # error = torch.mean(torch.norm(recon_error, dim=2))
                
                ## wav_src
                ref = torch.stack([torch.Tensor(gt_wav[machine_type]).unsqueeze(0) for machine_type in machine_types], dim=0)
                recon_sisdr = fast_bss_eval.si_sdr(ref[:, :, :, :pred_wav.shape[3]].cuda(), pred_wav)
                machine_idx = machine_types.index(machine_type)
                error = torch.mean(recon_sisdr[machine_idx, :, :])
                
                y_pred[num] = error.detach().cpu().numpy()
                if num <= 250:
                    for i, mt in enumerate(machine_types):
                        eval_types[mt].append(num)
                        error = torch.mean(recon_sisdr[i, :, :]).detach().cpu().numpy()
                        y_pred_types[mt][num] = error
                        y_true_types[mt][num] = y_true[num]
                else:
                    eval_types[machine_type].append(num)
                    y_pred_types[machine_type][num] = error.detach().cpu().numpy()
                    y_true_types[machine_type][num] = y_true[num]
            except:
                logger.warning("File broken!!: {}".format(file_name))

        scores = []
        for machine_type in machine_types:
            score = metrics.roc_auc_score(y_true_types[machine_type][eval_types[machine_type]], y_pred_types[machine_type][eval_types[machine_type]])
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
