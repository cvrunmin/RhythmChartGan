import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import time
import datetime

import gan_model
from gan_model import LossSensitiveGanLoss

# obtain by running calculate_prsk_dataset.py
group_class_weight = np.array([0.28247074844235354 * 0.75, 3.2944578026489815, 6.468063037130096, 600.7644882860666], dtype=np.float32)
color_class_weight = np.array([0.18831383229490234 * 0.75,
                               1.4767818026243325,
                               114.0082601116168,
                               447.7507696549189,
                               1314.3242514162396,
                               1246.8841970569417], dtype=np.float32)


def train_with_model(g_model: gan_model.ChartGanGen, d_model: gan_model.ChartGanDiss,
                     train_dataloader, device, epochs=5, lr=0.001, *, weight_ce=100, square_conv=True, use_stft=True):
    group_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(group_class_weight).to(device))
    color_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(color_class_weight).to(device))
    gan_loss_fn = LossSensitiveGanLoss(0)
    generator_optim = torch.optim.Adam(g_model.parameters(), lr=lr)
    discriminator_optim = torch.optim.Adam(d_model.parameters(), lr=lr)
    fix_chart = gan_model.RealChartFixer()
    softmax_chart = gan_model.GenChartFixer()

    if square_conv:
        window_length = 30
    else:
        window_length = 200

    d_losses = []
    g_losses = []
    full_img_loss = {'group': [], 'color': []}
    full_correct = {'group': [], 'color': []}

    for epoch in range(epochs):
        print('\n', "=" * 15, "Epoch", epoch + 1, "=" * 15)
        size = len(train_dataloader)
        g_model.train()
        d_model.train()
        train_img_loss = {'group': 0, 'color': 0}
        train_diss_loss = 0
        train_gen_loss = 0
        train_correct = {'group': 0, 'color': 0}
        correct_denominator_g = 0
        correct_denominator_c = 0
        print(f' {0}/{size}', end='\r', flush=True)
        start_time = time.process_time()
        for batch, pack in enumerate(train_dataloader):
            d_model.unlock()
            discriminator_optim.zero_grad()

            real_raw_img = pack['chart'].transpose(1, 0).to(device)

            real_img = fix_chart(real_raw_img)
            if use_stft:
                audio_feat = pack['lmel80'].transpose(1, 0).to(device)
            else:
                audio_feat = pack['cfp'].transpose(1, 0).to(device)
            level = pack['level'].transpose(1, 0).to(device)
            z = torch.normal(0, 1, size=[*level.shape, g_model.z_dim])
            z.requires_grad = False
            z = z.to(device)
            time_padding = 32 - 30 if square_conv else 256 - 200
            lane_padding = 32 - 12 if square_conv else 16 - 12
            real_padded_img = F.pad(real_img, [0, lane_padding, 0, time_padding])
            fake_padded_img, _ = g_model(z, level, audio_feat)
            fake_padded_img = fake_padded_img.detach()
            fake_padded_img = softmax_chart(fake_padded_img)
            fake_img = fake_padded_img[:, :, :, :window_length, :12]

            output_real = d_model(real_padded_img, level, audio_feat)
            output_fake = d_model(fake_padded_img, level, audio_feat)
            real_img = real_img.flatten(0, 1)
            fake_img = fake_img.flatten(0, 1)
            output_real = output_real.flatten()
            output_fake = output_fake.flatten()
            gan_loss = gan_loss_fn(real_img, fake_img, output_real, output_fake)
            gan_loss.backward()
            discriminator_optim.step()

            step_d_loss = gan_loss.detach().cpu().item()

            train_diss_loss += step_d_loss

            d_model.lock()
            generator_optim.zero_grad()

            z = torch.normal(0, 1, size=[*level.shape, g_model.z_dim])
            z.requires_grad = True
            z = z.to(device)
            fake_padded_img, _ = g_model(z, level, audio_feat)
            fake_img_group = fake_padded_img[:, :, :4, :window_length, :12].flatten(0,1)
            fake_img_color = fake_padded_img[:, :, 4:, :window_length, :12].flatten(0,1)
            fake_padded_img = softmax_chart(fake_padded_img)
            real_img_group = real_raw_img[:,:,1,:,:].flatten(0,1).long()
            real_img_color = real_raw_img[:,:,2,:,:].flatten(0,1).long()
            output_fake = d_model(fake_padded_img, level, audio_feat)
            output_fake = output_fake.mean()
            group_loss = group_loss_fn(fake_img_group, real_img_group) * weight_ce
            color_loss = color_loss_fn(fake_img_color, real_img_color) * weight_ce

            (group_loss + color_loss + output_fake).backward()
            generator_optim.step()

            step_g_loss = output_fake.detach().cpu().item()

            train_gen_loss += step_g_loss

            step_loss_g, step_loss_c = group_loss.detach().cpu().item(), color_loss.detach().cpu().item()
            step_correct_g = torch.logical_and(real_img_group != 0, fake_img_group.argmax(1) == real_img_group).detach().cpu().type(torch.float).sum().item()
            step_correct_c = torch.logical_and(real_img_color != 0, fake_img_color.argmax(1) == real_img_color).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_g = (real_img_group != 0).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_c = (real_img_color != 0).detach().cpu().type(torch.float).sum().item()
            train_img_loss['group'] += step_loss_g
            train_img_loss['color'] += step_loss_c
            train_correct['group'] += step_correct_g
            train_correct['color'] += step_correct_c
            correct_denominator_g += nonzero_tgt_g
            correct_denominator_c += nonzero_tgt_c
            step_acc_indicator_g = f'{(100*step_correct_g / nonzero_tgt_g):>0.1f}%' if nonzero_tgt_g != 0 else 'N/A'
            step_acc_indicator_c = f'{(100*step_correct_c / nonzero_tgt_c):>0.1f}%' if nonzero_tgt_c != 0 else 'N/A'

            print(f' {batch}/{size} Discriminator Loss: {step_d_loss:>4f}, Generator Loss: {step_g_loss:>4f},'+
                  f' image accuracy: {step_acc_indicator_g}, {step_acc_indicator_c},' +
                  f' image loss: {step_loss_g:>4f}, {step_loss_c:>4f}',
                  end='\r', flush=True)
        end_time = time.process_time()
        train_gen_loss /= size
        train_diss_loss /= size
        d_losses.append(train_diss_loss)
        g_losses.append(train_gen_loss)
        train_img_loss['group'] /= size
        train_img_loss['color'] /= size
        train_correct['group'] /= correct_denominator_g
        train_correct['color'] /= correct_denominator_c
        full_img_loss['group'].append(train_img_loss['group'])
        full_img_loss['color'].append(train_img_loss['color'])
        full_correct['group'].append(train_correct['group'])
        full_correct['color'].append(train_correct['color'])
        print(f' {datetime.timedelta(seconds=end_time - start_time)}' + 
            f' Train Avg Discriminator Loss: {train_diss_loss:>4f}, Avg Generator Loss: {train_gen_loss:>4f},' + 
            f" image accuracy: {(100 * train_correct['group']):>0.1f}%, {(100 * train_correct['color']):>0.1f}%," + 
            f" Avg image loss: {train_img_loss['group']:>8f}, {train_img_loss['color']:>8f}" + ' ' * 100)
    return d_losses, g_losses, full_correct, full_img_loss


def train_with_gen_only(g_model: gan_model.ChartGanGen, train_dataloader, device, epochs=5, lr=0.001):
    group_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(group_class_weight).to(device))
    color_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(color_class_weight).to(device))
    generator_optim = torch.optim.Adam(g_model.parameters(), lr=lr)
    full_img_loss = {'group': [], 'color': []}
    full_correct = {'group': [], 'color': []}


    for epoch in range(epochs):
        print('\n', "=" * 15, "Epoch", epoch + 1, "=" * 15)
        size = len(train_dataloader)
        g_model.train()
        train_img_loss = {'group': 0, 'color': 0}
        train_correct = {'group': 0, 'color': 0}
        correct_denominator_g = 0
        correct_denominator_c = 0
        print(f' {0}/{size}', end='\r', flush=True)
        start_time = time.process_time()
        for batch, pack in enumerate(train_dataloader):

            cfp = pack['cfp'].transpose(1, 0).to(device)
            level = pack['level'].transpose(1, 0).to(device)

            generator_optim.zero_grad()

            z = torch.normal(0, 1, size=[*level.shape, g_model.z_dim])
            z.requires_grad = True
            z = z.to(device)
            fake_padded_img, _ = g_model(z, level, cfp)
            fake_img_group = fake_padded_img[:, :, :4, :200, :12].flatten(0,1)
            fake_img_color = fake_padded_img[:, :, 4:, :200, :12].flatten(0,1)
            real_img_group = (pack['chart'].transpose(1, 0))[:,:,1,:,:].flatten(0,1).long().to(device)
            real_img_color = (pack['chart'].transpose(1, 0))[:,:,2,:,:].flatten(0,1).long().to(device)
            group_loss = group_loss_fn(fake_img_group, real_img_group)
            color_loss = color_loss_fn(fake_img_color, real_img_color)

            group_loss.backward(retain_graph=True)
            color_loss.backward()
            generator_optim.step()

            step_loss_g, step_loss_c = group_loss.detach().cpu().item(), color_loss.detach().cpu().item()
            step_correct_g = torch.logical_and(real_img_group != 0, fake_img_group.argmax(1) == real_img_group).detach().cpu().type(torch.float).sum().item()
            step_correct_c = torch.logical_and(real_img_color != 0, fake_img_color.argmax(1) == real_img_color).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_g = (real_img_group != 0).detach().cpu().type(torch.float).sum().item()
            nonzero_tgt_c = (real_img_color != 0).detach().cpu().type(torch.float).sum().item()
            train_img_loss['group'] += step_loss_g
            train_img_loss['color'] += step_loss_c
            train_correct['group'] += step_correct_g
            train_correct['color'] += step_correct_c
            correct_denominator_g += nonzero_tgt_g
            correct_denominator_c += nonzero_tgt_c
            step_acc_indicator_g = f'{(100*step_correct_g / nonzero_tgt_g):>0.1f}%' if nonzero_tgt_g != 0 else 'N/A'
            step_acc_indicator_c = f'{(100*step_correct_c / nonzero_tgt_c):>0.1f}%' if nonzero_tgt_c != 0 else 'N/A'

            print(f' {batch}/{size} accuracy: {step_acc_indicator_g}, {step_acc_indicator_c},' +
                  f' loss: {step_loss_g:>4f}, {step_loss_c:>4f}' + ' '*100, end='\r',
                  flush=True)
        end_time = time.process_time()
        train_img_loss['group'] /= size
        train_img_loss['color'] /= size
        train_correct['group'] /= correct_denominator_g
        train_correct['color'] /= correct_denominator_c
        full_img_loss['group'].append(train_img_loss['group'])
        full_img_loss['color'].append(train_img_loss['color'])
        full_correct['group'].append(train_correct['group'])
        full_correct['color'].append(train_correct['color'])
        print(f" {datetime.timedelta(seconds=end_time - start_time)} Train accuracy:" +
              f" {(100 * train_correct['group']):>0.1f}%," +
              f" {(100 * train_correct['color']):>0.1f}%, Avg loss: {train_img_loss['group']:>8f}, {train_img_loss['color']:>8f}")
    return full_correct, full_img_loss


def level_filter_stage_1(level):
    return level <= 9
def level_filter_stage_2(level):
    return level <= 14
def level_filter_stage_3(level):
    return level <= 20
def level_filter_stage_4(level):
    return level <= 27
def level_filter_stage_5(level):
    return True
level_predicates = [level_filter_stage_1, level_filter_stage_2, level_filter_stage_3, level_filter_stage_4, level_filter_stage_5]


def main():
    import argparse
    import torch
    import dataset
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='prsk_dataset')
    parser.add_argument('--chkpt_dir', type=str, default='.')
    parser.add_argument('--chkpt_fmt', type=str, default='chartgen_{gd}_stage_{stage}.pth')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--end_stage',type=int, default=5)
    parser.add_argument('--start_stage', type=int, default=0)
    parser.add_argument('--use_cfp', action='store_true')
    parser.add_argument('--long_stride', action='store_true')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--ce_weight', type=float, default=80)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if args.use_cfp:
        audio_extractor = gan_model.CfpFeatureSubmodule()
    else:
        audio_extractor = gan_model.MelFeatureSubmodule()
    if args.long_stride:
        square_conv = False
        window_len = 200
    else:
        square_conv = True
        window_len = 30

    chkpt_dir = Path(args.chkpt_dir)
    if not chkpt_dir.is_dir():
        chkpt_dir.mkdir(parents=True)

    model_g = gan_model.ChartGanGen(audio_feature_module=audio_extractor, window_length=window_len, square_conv=square_conv).to(device)
    model_d = gan_model.ChartGanDiss(model_g.level_embedder, model_g.cfp_feature_extractor, window_length=window_len, square_conv=square_conv).to(device)
    if args.start_stage > 0:
        if (Path(args.chkpt_dir) / args.chkpt_fmt.format(gd='g', stage=args.start_stage)).is_file():
            model_g.load_state_dict(torch.load(str(chkpt_dir / args.chkpt_fmt.format(gd='g', stage=args.start_stage))))
        else:
            print('WARN: start training G at middle without checkpoint')
        if (Path(args.chkpt_dir) / args.chkpt_fmt.format(gd='d', stage=args.start_stage)).is_file():
            model_g.load_state_dict(torch.load(str(chkpt_dir / args.chkpt_fmt.format(gd='d', stage=args.start_stage))))
        else:
            print('WARN: start training D at middle without checkpoint')
    for stage in range(args.start_stage, args.end_stage):
        level_predicate = level_predicates[min(stage, 4)]
        full_dataset = dataset.ChartGenSequenceDataset(args.data_dir, window_length=window_len, window_step=window_len, seq_len=args.seq_len,level_predicate=level_predicate)
        dataset_len = len(full_dataset)
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [round(dataset_len * 0.7), dataset_len - round(dataset_len * 0.7)], torch.Generator().manual_seed(0))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        d_losses, g_losses, img_acc, img_loss = train_with_model(model_g, model_d, train_dataloader, device,
             epochs=args.num_epochs, lr=args.lr, weight_ce=args.ce_weight, square_conv=square_conv, use_stft=not args.use_cfp)
        torch.save(model_g.state_dict(), chkpt_dir / args.chkpt_fmt.format(gd='g', stage=stage+1))
        torch.save(model_d.state_dict(), chkpt_dir / args.chkpt_fmt.format(gd='d', stage=stage+1))
        print(f'Stage {stage+1} done.')
        print('-'*20)



if __name__ == '__main__':
    main()