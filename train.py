import argparse
import pandas as pd
import numpy as np
import torch
import os
import re

from models import GraphConvolution, GraphConvolutionalEncoder, GRACE, learner
from utils import setup_config_args, fix_seed, get_logger, get_activation, save_np, save_heatmap, make_route, get_device
from functionals import symmetric_normalization, normalization, similarity_matrix, ssm_fusion, concatenate_fusion

from dataloader.dataloader import AudioTextDataset, load_dataset
from dataloader.dataembedding import DataProcessor


def get_dataset(args):
    file_path = args.data_load_path
    audio_dir = args.audio_dir

    dataset = load_dataset(file_path, audio_dir)

    stt_path = os.path.join(os.path.dirname(file_path), 'stt_results.csv')
    processor = DataProcessor(dataset, stt_path, args)

    data = processor.process()

    if 'filename' in data.columns:
        subjects = data['filename'].apply(lambda x: str(x).split('_')[0]).to_numpy()
        subject_ids = np.array([re.findall(r'\d+', str(s))[0] for s in subjects])
    else:
        print("Warning: 'filename' column not found. Using dummy IDs.")
        subject_ids = np.zeros(len(data))

    label = torch.from_numpy(data.loc[:, 'emotion_label'].to_numpy()).to(torch.long)

    audio = torch.from_numpy(
        data.loc[:, 'audio_feature1':'audio_feature' + str(args.n_audio_features)].to_numpy()).to(
        torch.float32) if 'a' in args.modality else torch.empty(0)

    text = torch.from_numpy(data.loc[:, 'text_feature1':].to_numpy()).to(
        torch.float32) if 't' in args.modality else torch.empty(0)

    label = torch.where(torch.isnan(label), torch.zeros_like(label), label)
    audio = torch.where(torch.isnan(audio), torch.zeros_like(audio), audio)
    text = torch.where(torch.isnan(text), torch.zeros_like(text), text)

    return label, audio, text, subject_ids


def main(args):

    fix_seed(args.seed)

    file_name = args.dataset + '.log'
    make_route(args.log_save_path, args.dataset + '.log')
    filepath = os.path.join(args.log_save_path, file_name)
    log = get_logger(filepath=filepath)

    log.info(args)

    device = get_device(args.cuda_id)
    log.info(f'Using device: {device}')

    label, audio, text, subject_ids = get_dataset(args)

    unique_subjects = np.unique(subject_ids)
    num_subjects = len(unique_subjects)

    log.info(f'Label shape : {label.shape}')
    log.info(f'Subject IDs shape: {subject_ids.shape}')
    log.info(f'Unique Subjects ({num_subjects}): {unique_subjects}')

    im = torch.eye(len(subject_ids), dtype=torch.float32)

    total_avg_list, total_std_list, total_cfm_list = np.array([]), np.array([]), np.array([])
    total_macro_f1_list, total_weighted_f1_list = np.array([]), np.array([])

    date = '240321'

    if audio.nelement() > 0:
        audio = normalization(audio, axis=0, ntype='standardization')
    if text.nelement() > 0:
        text = normalization(text, axis=0, ntype='standardization')

    for j in range(args.n_times_draw):
        class_f1_score = [0] * 7
        best_acc_list, cfm_list, macro_f1_list, weighted_f1_list = [], [], [], []

        log.info(
            "=================================== Iteration {} ======================================".format(j + 1))

        np.random.shuffle(unique_subjects)
        subject_folds = np.array_split(unique_subjects, args.n_folds)

        for i in range(args.n_folds):
            log.info("******************* TEST fold 3:1 / Fold {} *********************".format(i + 1))

            log.info("similarity matrix construction start...")
            if 'a' in args.modality:
                asm = similarity_matrix(audio, scale=0.9)
                sasm = (asm + asm.T) / 2
                nsasm = symmetric_normalization(sasm, im)

            if 't' in args.modality:
                tsm = similarity_matrix(text, scale=0.9)
                stsm = (tsm + tsm.T) / 2
                nstsm = symmetric_normalization(stsm, im)

            fsm = None
            if args.modality == 'at':
                fsm = ssm_fusion(asm, nsasm, tsm, nstsm, args.k_neighbor, args.timestep)
            elif args.modality == 'a':
                fsm = nsasm
            elif args.modality == 't':
                fsm = nstsm

            log.info("similarity matrix construction Done...")

            # Heatmap 저장
            make_route(args.structure_save_path)

            ffm = None
            if args.modality == 'at':
                ffm = concatenate_fusion(audio, text)
            elif args.modality == 'a':
                ffm = audio
            elif args.modality == 't':
                ffm = text

            adj = fsm.to(device)
            feature = ffm.to(device)
            label = label.to(device)

            log.info(f"graph node features are saved to {args.feature_save_path}")

            # =================================================================
            # Subject-Independent Masking
            # =================================================================
            test_target_subjects = subject_folds[i]
            all_subjects_set = set(unique_subjects)
            test_subjects_set = set(test_target_subjects)
            train_target_subjects = np.array(list(all_subjects_set - test_subjects_set))

            is_test_mask = np.isin(subject_ids, test_target_subjects)

            test_identifier = torch.from_numpy(is_test_mask).bool().to(device)
            train_identifier = ~test_identifier

            n_train = train_identifier.sum().item()
            n_test = test_identifier.sum().item()

            log.info(f"Fold {i + 1} Split -> Train Samples: {n_train}, Test Samples: {n_test}")

            log.info(f"Test Subjects ({len(test_target_subjects)}): {test_target_subjects}")
            log.info(f"Train Subjects ({len(train_target_subjects)}): {train_target_subjects}")

            intersection = test_subjects_set.intersection(set(train_target_subjects))
            if intersection:
                log.error(f"CRITICAL WARNING: Data Leakage Detected! Overlapping Subjects: {intersection}")
                raise ValueError("Train and Test sets share the same subjects!")
            # =================================================================

            activation = get_activation('celu')
            gcn = GraphConvolution

            encoder = GraphConvolutionalEncoder(args.n_features, args.gcn_hid_channels, args.gcn_out_channels,
                                                activation,
                                                base_model=gcn).to(device)
            model = GRACE(encoder, args.n_features, args.gcn_out_channels, args.proj_hid_channels, args.out_channels,
                          args.ptau).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            model_id = 'iteration' + str(j + 1) + '-fold' + str(i + 1)

            make_route(args.model_save_path)
            best_acc, best_z, cfm, report, out_trigger, best_epoch = learner(
                model_id, model, optimizer, feature, adj, label,
                train_identifier, test_identifier,
                args, isdeap=False, verbose=False, earlystop=False
            )

            # 로그 및 결과 저장
            if out_trigger == 0:
                log.info('model id: {}, Epoch : {} - Find best epoch'.format(model_id, best_epoch))
            elif out_trigger == 1:
                log.info('model id: {}, Epoch : {} - Accuracy - 100.%'.format(model_id, best_epoch))
            elif out_trigger == 2:
                log.info('model id: {}, Epoch : {} - Early Stopping'.format(model_id, best_epoch))

            model_save_file = args.model_save_path + 'subject_independent_' + model_id + '.pt'
            torch.save(model.state_dict(), model_save_file)

            best_acc_list.append(best_acc.item())

            if len(cfm_list) == 0:
                cfm_list = cfm[np.newaxis]
            else:
                cfm_list = np.concatenate([cfm_list, cfm[np.newaxis]])

            weighted_f1_list.append(report['weighted avg']['f1-score'])
            macro_f1_list.append(report['macro avg']['f1-score'])

            # Class별 F1 Score 합산
            target_names = ['기쁨', '중립', '불안', '당황', '상처', '슬픔', '분노']
            for k, name in enumerate(target_names):
                if name in report:
                    class_f1_score[k] += report[name]['f1-score']

        # Iteration별 통계
        avg = np.mean(best_acc_list)
        std = np.std(best_acc_list)
        log.info(f"** Iteration {j + 1} Best ACC: {np.max(best_acc_list):.2f}, Avg: {avg:.2f}, Std: {std:.2f} **\n")

        total_avg_list = np.append(total_avg_list, avg)
        total_std_list = np.append(total_std_list, std)

        if len(total_cfm_list) == 0:
            total_cfm_list = np.mean(cfm_list, axis=0, keepdims=True)
        else:
            total_cfm_list = np.concatenate([total_cfm_list, np.mean(cfm_list, axis=0, keepdims=True)])

        total_weighted_f1_list = np.append(total_weighted_f1_list, np.mean(weighted_f1_list))
        total_macro_f1_list = np.append(total_macro_f1_list, np.mean(macro_f1_list))

    # 최종 통계 저장
    log.info(f"Final Avg ACC: {np.mean(total_avg_list):.2f}")

    make_route(args.tensor_save_path)
    save_np(args.tensor_save_path, 'total_avg_' + date, total_avg_list)
    save_np(args.tensor_save_path, 'total_std_' + date, total_std_list)
    save_np(args.tensor_save_path, 'avg_cfm_' + date, np.mean(total_cfm_list, axis=0))
    save_np(args.tensor_save_path, 'total_weighted_f1_' + date, total_weighted_f1_list)
    save_np(args.tensor_save_path, 'total_macro_f1_' + date, total_macro_f1_list)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments set from prompt and YAML configuration.')
    parser.add_argument('--dataset', type=str, default='IITP-SMED-ORIGIN', help='dataset (default: IITP-SMED)')
    parser.add_argument('--cuda_id', type=str, default='0', help='Cuda device ID (default: 0)')
    args = setup_config_args(parser, filepath='config.yaml')

    main(args)
