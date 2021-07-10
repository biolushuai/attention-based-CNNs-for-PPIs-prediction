import os
import time

import pickle
import numpy as np
import torch
import torch.utils.data.sampler as sampler

from utils import *
from models.LocalCNN import CNNPPI
from models.LocalCNNAttention import AttentionPPI
from data_generator import ProteinDataSet
from evaluation import *
from config import DefaultConfig
configs = DefaultConfig()
THREADHOLD = 0.2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_epoch(model, loader, optimizer, epoch, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()

    epochs = configs.epochs

    global THREADHOLD
    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (seq_data, pssm_data, dssp_data, rasa_data, angle_data, local_data, label) in enumerate(loader):
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                seq_var = seq_data.cuda().float()
                pssm_var = pssm_data.cuda().float()
                dssp_var = dssp_data.cuda().float()
                rasa_var = rasa_data.cuda().float()
                angle_var = angle_data.cuda().float()
                local_var = local_data.cuda().float()
                target_var = label.cuda().float()
            else:
                seq_var = seq_data.float()
                pssm_var = pssm_data.float()
                dssp_var = dssp_data.float()
                rasa_var = rasa_data.float()
                angle_var = angle_data.float()
                local_var = local_data.float()
                target_var = label.float()

        # compute output
        # print(seq_var.shape, pssm_var.shape, dssp_var.shape, local_var.shape)
        output = model(local_var)
        shapes = output.data.shape
        output = output.view(shapes[0]*shapes[1])
        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()

        # measure accuracy and record loss
        batch_size = label.size(0)
        pred_out = output.ge(THREADHOLD)

        MiP, MiR, MiF, PNum, RNum = micro_score(pred_out.data.cpu().numpy(), target_var.data.cpu().numpy())
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'f_max:%.6f' % (MiP),
                'p_max:%.6f' % (MiR),
                'r_max:%.6f' % (MiF),
                't_max:%.2f' % (PNum)])
            print(res)

    return batch_time.avg, losses.avg


def eval_epoch(model, loader, print_freq=10, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()

    global THREADHOLD
    # Model on eval mode
    model.eval()

    all_trues = []
    all_preds = []
    end = time.time()
    for batch_idx, (seq_data, pssm_data, dssp_data, rasa_data, angle_data, local_data, label) in enumerate(loader):

        # Create variables
        with torch.no_grad():
            if torch.cuda.is_available():
                seq_var = seq_data.cuda().float()
                pssm_var = pssm_data.cuda().float()
                dssp_var = dssp_data.cuda().float()
                rasa_var = rasa_data.cuda().float()
                angle_var = angle_data.cuda().float()
                local_var = local_data.cuda().float()
                target_var = label.cuda().float()
            else:
                seq_var = seq_data.float()
                pssm_var = pssm_data.float()
                dssp_var = dssp_data.float()
                rasa_var = rasa_data.float()
                angle_var = angle_data.float()
                local_var = local_data.float()
                target_var = label.float()

        # compute output
        output = model(local_var)
        shapes = output.data.shape
        output = output.view(shapes[0]*shapes[1])

        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()

        # measure accuracy and record loss
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            ])
            print(res)
        all_trues.append(label.numpy())
        all_preds.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds, all_trues)
    acc_val = acc_score(predictions_max, all_trues)
    mcc = compute_mcc(predictions_max, all_trues)
    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc


def train(model, train_data_set, save, seed=None, num=1, train_file=None):
    batch_size = configs.batch_size
    lr = configs.learning_rate
    if seed is not None:
        torch.manual_seed(seed)

    global THREADHOLD

    # split data
    with open(train_file, "rb") as fp:
        train_list = pickle.load(fp)

    samples_num = len(train_list)
    split_num = int(configs.split_rate * samples_num)
    data_index = train_list
    np.random.shuffle(data_index)
    train_index = data_index[:split_num]
    eval_index = data_index[split_num:]
    train_samples = sampler.SubsetRandomSampler(train_index)
    eval_samples = sampler.SubsetRandomSampler(eval_index)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=train_samples,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=eval_samples,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model

    # Optimizer
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=lr)

    # Start log
    with open(os.path.join(save, 'AttentionPPI_results{}.csv'.format(num)), 'w') as f:
        f.write('epoch, loss, acc, F_value, precision, recall, auc, aupr, mcc, threadhold\n')

        # Train model
        best_F = 0
        threadhold = 0
        count = 0
        for epoch in range(epochs):
            _, train_loss = train_epoch(model=model_wrapper, loader=train_loader, optimizer=optimizer, epoch=epoch)
            _, valid_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = eval_epoch(
                model=model_wrapper,
                loader=valid_loader,
                is_test=(not valid_loader)
            )
            # Log results
            f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n' % (
            (epoch + 1), valid_loss, acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))
            print(
                'epoch:%03d,valid_loss:%0.5f\nacc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f\n' % (
                    (epoch + 1), valid_loss, acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))
            if f_max > best_F:
                count = 0
                best_F = f_max
                THREADHOLD = t_max
                print("new best F_value:{0}(threadhold:{1})".format(f_max, THREADHOLD))
                torch.save(model.state_dict(), os.path.join(save, 'AttentionPPI_model{}.dat'.format(num)))
            else:
                count += 1
                if count >= 5:
                    return None


if __name__ == '__main__':
    # para
    batch_size = configs.batch_size
    epochs = configs.epochs
    save_path = configs.save_path
    train_data_sets = configs.data_sets

    train_sequences_file = ['data_cache/{0}_sequence_data.pkl'.format(key) for key in train_data_sets]
    train_dssp_file = ['data_cache/{0}_dssp2_data.pkl'.format(key) for key in train_data_sets]
    train_pssm_file = ['data_cache/{0}_pssm_data.pkl'.format(key) for key in train_data_sets]
    train_rasa_file = ['data_cache/{0}_rasa_data.pkl'.format(key) for key in train_data_sets]
    train_angle_file = ['data_cache/{0}_angle_data.pkl'.format(key) for key in train_data_sets]
    train_label_file = ['data_cache/{0}_label.pkl'.format(key) for key in train_data_sets]
    all_list_file = 'data_cache/all_dset_list.pkl'
    train_list_file = 'data_cache/training_list.pkl'

    # Datasets
    train_dataset = ProteinDataSet(train_sequences_file, train_pssm_file, train_dssp_file, train_rasa_file, train_angle_file, train_label_file, all_list_file)

    # Models
    class_nums = 1
    # model = CNNPPI(class_nums)
    model = AttentionPPI(class_nums)
    model.apply(weight_init)

    # Train the model
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # seeds = [649737, 395408, 252356, 343053, 743746]
    seeds = [395408]

    for i, seed in enumerate(seeds):
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        train(model=model, train_data_set=train_dataset, save=save_path, num=i + 1, train_file=train_list_file)
        print('Done!')
