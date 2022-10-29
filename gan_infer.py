import torch
import torch.nn.functional as F
import gan_model
from gan_model import ChartGanGen, ChartMaker
from typing import Tuple, Union
from pathlib import Path
import utils
import numpy as np


def infer_chart(model: ChartGanGen, audio_path: Union[str, Path], tgt_level, device, max_length=-1, *, window_length=200, seq_len=5, use_stft=False) -> Tuple[torch.Tensor]:
    if use_stft:
        audio_feat_raw = utils.get_lmel80(audio_path)
    else:
        audio_feat_raw, _, _ = utils.get_cfp(audio_path, hop=None, fps=50, model_type='melody')
    audio_feat_full = torch.from_numpy(audio_feat_raw).float().to(device)
    if max_length != -1:
        audio_feat_full = audio_feat_full[:, :, :max_length]
    audio_feat_full = F.pad(audio_feat_full, [0, -audio_feat_full.size(-1) % window_length])
    level_tensor = torch.tensor([[tgt_level]] * seq_len).to(device)
    ideal_length = audio_feat_full.size(-1) // window_length
    curr_predict_length = 0
    initial_h = None
    generated_img = []
    print(f'Progress: 0/{ideal_length}', end='\r', flush=True)
    with torch.no_grad():
        while curr_predict_length < ideal_length:
            cfp = audio_feat_full[:,:,window_length * curr_predict_length:window_length * (curr_predict_length + 5)]
            cfp = cfp.reshape(-1, 1, *audio_feat_full.shape[0:2], window_length)
            lvl = level_tensor[:cfp.size(0), :]  # trim sequence length to match cfp
            z = torch.normal(0, 1, size=[*lvl.shape, model.z_dim]).to(device)
            fake_padded_img, h = model(z, lvl, cfp, initial_h=initial_h)
            fake_img: torch.Tensor
            fake_img = fake_padded_img[:, :, :, :window_length, :12].squeeze(1)
            fake_img = fake_img.detach().cpu()
            generated_img.extend(fake_img.unbind(0))  # break tensor into sequences
            initial_h = h
            curr_predict_length += lvl.size(0)
            print(f'Progress: {curr_predict_length}/{ideal_length}', end='\r', flush=True)
    hot_chart = torch.cat(generated_img, dim=1)
    maker = ChartMaker()
    return maker(hot_chart)


def main():
    import argparse
    import torch
    import os.path
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--chkpt_path', type=str, required=True)
    parser.add_argument('-f', '--audio_path', type=str, required=True)
    parser.add_argument('-l', '--level', type=int, required=True)
    parser.add_argument('--use_cfp', dest='use_stft', action='store_false')
    parser.add_argument('--long_stride', action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if args.use_stft:
        audio_extractor = gan_model.MelFeatureSubmodule()
    else:
        audio_extractor = gan_model.CfpFeatureSubmodule()
    if args.long_stride:
        square_conv = False
        window_len = 200
    else:
        square_conv = True
        window_len = 30

    model_g = gan_model.ChartGanGen(audio_feature_module=audio_extractor, window_length=window_len, square_conv=square_conv).to(device)
    model_g.load_state_dict(torch.load(args.chkpt_path))

    chart = np.stack(infer_chart(model_g, args.audio_path, args.level, device=device, window_length=window_len, use_stft=args.use_stft), axis=0)
    out_dir, fullname = os.path.split(args.audio_path)
    fname, _ = os.path.splitext(fullname)
    np.save(os.path.join(out_dir, fname + '_chart'), chart)
    print(f'saved chart into {os.path.join(out_dir, fname + "_chart.npy")}')
    



if __name__ == '__main__':
    main()