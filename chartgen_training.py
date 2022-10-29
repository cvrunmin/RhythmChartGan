import numpy as np
import torch
from torch import nn

# obtain by running calculate_prsk_dataset.py
alpha_class_weight = np.array([0.552931410134545 * 0.75, 5.223093516014989], dtype=np.float32)
group_class_weight = np.array([0.28247074844235354 * 0.75, 3.2944578026489815, 6.468063037130096, 600.7644882860666], dtype=np.float32)
color_class_weight = np.array([0.18831383229490234 * 0.75,
                               1.4767818026243325,
                               114.0082601116168,
                               447.7507696549189,
                               1314.3242514162396,
                               1246.8841970569417], dtype=np.float32)


def train_with_model(model, train_dataloader, device, epochs=5):
    """
    :param model: model to be trained
    :param train_dataloader:
    :param device: device of model
    :param epochs:
    :return:
    """
    alpha_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(alpha_class_weight).to(device))
    group_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(group_class_weight).to(device))
    color_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(color_class_weight).to(device))
    # alpha_loss_fn = nn.CrossEntropyLoss()
    # group_loss_fn = nn.CrossEntropyLoss()
    # color_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    for t in range(epochs):
        print('\n', "=" * 15, "Epoch", t + 1, "=" * 15)
        size = len(train_dataloader)
        model.train()
        train_loss = {'alpha': 0, 'group': 0, 'color': 0}
        train_correct = {'alpha': 0, 'group': 0, 'color': 0}
        correct_denominator_a = 0
        correct_denominator_g = 0
        correct_denominator_c = 0
        print(f' {0}/{size}', end='\r', flush=True)
        for batch, pack in enumerate(train_dataloader):
            melody = pack['cfp'].float().to(device)
            chart = pack['chart'].float().to(device)
            level = pack['level'].int().to(device)
            pred_alpha, pred_group, pred_color = model(melody, chart, level)
            tgt_alpha, tgt_group, tgt_color = chart.long().split(1, dim=1)
            tgt_alpha = tgt_alpha.squeeze(1)
            tgt_group = tgt_group.squeeze(1)
            tgt_color = tgt_color.squeeze(1)
            alpha_loss = alpha_loss_fn(pred_alpha.reshape(-1, pred_alpha.shape[-1]), tgt_alpha.reshape(-1))
            group_loss = group_loss_fn(pred_group.reshape(-1, pred_group.shape[-1]), tgt_group.reshape(-1))
            color_loss = color_loss_fn(pred_color.reshape(-1, pred_color.shape[-1]), tgt_color.reshape(-1))

            optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            group_loss.backward(retain_graph=True)
            color_loss.backward()
            optimizer.step()

            step_loss_a, step_loss_g, step_loss_c = alpha_loss.detach().cpu().item(), \
                group_loss.detach().cpu().item(), color_loss.detach().cpu().item()
            step_correct_a = torch.logical_and(tgt_alpha != 0, pred_alpha.argmax(-1) == tgt_alpha).detach().cpu().type(torch.float).sum().item()
            step_correct_g = torch.logical_and(tgt_group != 0, pred_group.argmax(-1) == tgt_group).detach().cpu().type(torch.float).sum().item()
            step_correct_c = torch.logical_and(tgt_color != 0, pred_color.argmax(-1) == tgt_color).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_a = (tgt_alpha != 0).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_g = (tgt_group != 0).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_c = (tgt_color != 0).detach().cpu().type(torch.float).sum().item()
            train_loss['alpha'] += step_loss_a
            train_loss['group'] += step_loss_g
            train_loss['color'] += step_loss_c
            train_correct['alpha'] += step_correct_a
            train_correct['group'] += step_correct_g
            train_correct['color'] += step_correct_c
            correct_denominator_a += nonzero_tgt_a
            correct_denominator_g += nonzero_tgt_g
            correct_denominator_c += nonzero_tgt_c
            step_acc_indicator_a = f'{(100*step_correct_a / nonzero_tgt_a):>0.1f}%' if nonzero_tgt_a != 0 else 'N/A'
            step_acc_indicator_g = f'{(100*step_correct_g / nonzero_tgt_g):>0.1f}%' if nonzero_tgt_g != 0 else 'N/A'
            step_acc_indicator_c = f'{(100*step_correct_c / nonzero_tgt_c):>0.1f}%' if nonzero_tgt_c != 0 else 'N/A'
            print(f' {batch}/{size} accuracy: {step_acc_indicator_a}, {step_acc_indicator_g}, {step_acc_indicator_c},' +
                  f' loss: {step_loss_a:>4f}, {step_loss_g:>4f}, {step_loss_c:>4f}' + ' '*100, end='\r',
                  flush=True)
        train_loss['alpha'] /= size
        train_loss['group'] /= size
        train_loss['color'] /= size
        train_correct['alpha'] /= correct_denominator_a
        train_correct['group'] /= correct_denominator_g
        train_correct['color'] /= correct_denominator_c
        print(f" Train accuracy: {(100 * train_correct['alpha']):>0.1f}%," +
              f" {(100 * train_correct['group']):>0.1f}%," +
              f" {(100 * train_correct['color']):>0.1f}%, Avg loss: {train_loss['alpha']:>8f}, {train_loss['group']:>8f}, {train_loss['color']:>8f}")


def test_model(model, test_dataloader, device):
    """
    :param model: model to be trained
    :param train_dataloader:
    :param device: device of model
    :param epochs:
    :return:
    """
    alpha_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(alpha_class_weight).to(device))
    group_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(group_class_weight).to(device))
    color_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(color_class_weight).to(device))
    # alpha_loss_fn = nn.CrossEntropyLoss()
    # group_loss_fn = nn.CrossEntropyLoss()
    # color_loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        size = len(test_dataloader)
        model.eval()
        test_loss = {'alpha': 0, 'group': 0, 'color': 0}
        test_correct = {'alpha': 0, 'group': 0, 'color': 0}
        correct_denominator_a = 0
        correct_denominator_g = 0
        correct_denominator_c = 0
        print(f' {0}/{size}', end='\r', flush=True)
        for batch, pack in enumerate(test_dataloader):
            melody = pack['cfp'].float().to(device)
            chart = pack['chart'].float().to(device)
            level = pack['level'].int().to(device)
            pred_alpha, pred_group, pred_color = model(melody, chart, level)
            tgt_alpha, tgt_group, tgt_color = chart.long().split(1, dim=1)
            tgt_alpha = tgt_alpha.squeeze(1)
            tgt_group = tgt_group.squeeze(1)
            tgt_color = tgt_color.squeeze(1)
            alpha_loss = alpha_loss_fn(pred_alpha.reshape(-1, pred_alpha.shape[-1]), tgt_alpha.reshape(-1))
            group_loss = group_loss_fn(pred_group.reshape(-1, pred_group.shape[-1]), tgt_group.reshape(-1))
            color_loss = color_loss_fn(pred_color.reshape(-1, pred_color.shape[-1]), tgt_color.reshape(-1))

            step_loss_a, step_loss_g, step_loss_c = alpha_loss.detach().cpu().item(), \
                group_loss.detach().cpu().item(), color_loss.detach().cpu().item()
            step_correct_a = torch.logical_and(tgt_alpha != 0, pred_alpha.argmax(-1) == tgt_alpha).detach().cpu().type(torch.float).sum().item()
            step_correct_g = torch.logical_and(tgt_group != 0, pred_group.argmax(-1) == tgt_group).detach().cpu().type(torch.float).sum().item()
            step_correct_c = torch.logical_and(tgt_color != 0, pred_color.argmax(-1) == tgt_color).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_a = (tgt_alpha != 0).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_g = (tgt_group != 0).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_c = (tgt_color != 0).detach().cpu().type(torch.float).sum().item()
            test_loss['alpha'] += step_loss_a
            test_loss['group'] += step_loss_g
            test_loss['color'] += step_loss_c
            test_correct['alpha'] += step_correct_a
            test_correct['group'] += step_correct_g
            test_correct['color'] += step_correct_c
            correct_denominator_a += nonzero_tgt_a
            correct_denominator_g += nonzero_tgt_g
            correct_denominator_c += nonzero_tgt_c
            print(f' {batch}/{size}', end='\r', flush=True)
        test_loss['alpha'] /= size
        test_loss['group'] /= size
        test_loss['color'] /= size
        test_correct['alpha'] /= correct_denominator_a
        test_correct['group'] /= correct_denominator_g
        test_correct['color'] /= correct_denominator_c
        print(f" Test accuracy: {(100 * test_correct['alpha']):>0.1f}%," +
              f" {(100 * test_correct['group']):>0.1f}%," +
              f" {(100 * test_correct['color']):>0.1f}%, Avg loss: {test_loss['alpha']:>8f}, {test_loss['group']:>8f}, {test_loss['color']:>8f}")