import torch
import torch.nn.functional as F
from ar_model import ChartGenModel
from pathlib import Path
from typing import Union
import utils


def get_device(force_cpu=False):
    return "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"


def infer_chart(model: ChartGenModel, audio_path: Union[str, Path], tgt_level, device, max_length=-1):
    window_length = 500
    window_step = 250

    audio_cfp, _, _ = utils.get_cfp(audio_path, hop=None, fps=50, model_type='melody')
    cfp_tensor = torch.from_numpy(audio_cfp).float().unsqueeze(0).to(device)
    melody = cfp_tensor.permute(0, 1, 3, 2)
    if max_length != -1:
        melody = melody[:, :, :max_length, :]
    max_frame = melody.shape[-2]
    level_tensor = torch.tensor([tgt_level]).to(device)
    sample_tensor = torch.zeros([1, 1, 1, 1])
    ideal_length = max_frame * 36
    initial_sample = True
    curr_predict_length = 0
    curr_output_length = 0
    curr_frame_start = 0
    print(f'Progress: 0/{ideal_length}', end='\r', flush=True)
    with torch.no_grad():
        while curr_predict_length < ideal_length:
            chart_tensor = sample_tensor[:, :, curr_frame_start * 36:(curr_frame_start + window_length) * 36, :].to(
                device)
            output: torch.Tensor
            output = model(melody[:, curr_frame_start:curr_frame_start + window_length],
                           chart_tensor,
                           level_tensor, predict=True)
            output = output[:, :, curr_output_length:curr_output_length + 1, :].detach().cpu()
            if initial_sample:
                sample_tensor = output.clone()
                initial_sample = False
            else:
                sample_tensor = torch.cat([sample_tensor, output], dim=2)
            curr_predict_length += 1
            curr_output_length += 1
            print(f'Progress: {curr_predict_length}/{ideal_length}', end='\r', flush=True)
            if curr_output_length == 36 * window_length:
                if curr_frame_start + window_step + window_length > max_frame:
                    curr_frame_start = max_frame - window_length
                else:
                    curr_frame_start += window_step
                curr_output_length = curr_predict_length - curr_frame_start * 36
    chart = torch.stack(ChartGenModel.uninterleave_charts(sample_tensor.reshape([1, -1, 36])), 1)
    return chart.squeeze(0)
