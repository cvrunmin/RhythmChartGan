import torch
import math
from torch import Tensor, nn
from torch.nn import functional as F
from modules import ImageTransformerDecoder, ImageTransformerDecoder1dLayer


def add_timing_signal(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4):
    '''
    PyTorch version of add_timing_signal_nd from tensor2tensor/common_attention
    '''
    num_dims = len(x.shape) - 2
    channels = x.shape[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1)
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales) * -log_timescale_increment
    )
    inv_timescales = inv_timescales.to(x.device)
    for dim in range(num_dims):
        length = x.shape[dim + 1]
        position = torch.arange(length).float().to(x.device)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.concat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = F.pad(signal, [prepad, postpad])
        for _ in range(1 + dim):
            signal = signal.unsqueeze(0)
        for _ in range(num_dims - 1 - dim):
            signal = signal.unsqueeze(-2)
        x += signal
    return x


def right_shift_blockwise(x, query_shape):
    from utils.attn_utils import pad_to_multiple_2d, gather_indices_2d, get_shifted_center_blocks, scatter_blocks_2d
    '''
    PyTorch version of right_shift_blockwise from tensor2tensor/common_attention
    '''
    x_shape = x.shape
    x = x.unsqueeze(1)
    x = pad_to_multiple_2d(x, query_shape)
    padded_x_shape = x.shape
    x_indices = gather_indices_2d(x, query_shape, query_shape)
    x_new = get_shifted_center_blocks(x, x_indices)

    output = scatter_blocks_2d(x_new, x_indices, padded_x_shape)
    output = output.squeeze(1)
    output = output[:, :x_shape[1], :x_shape[2], :]
    return output


def right_shift_3d(x):
    return F.pad(x, [0, 0, 1, 0])[:, :-1, :]


class AudioFeatureModule(nn.Module):
    def __init__(self):
        super(AudioFeatureModule, self).__init__()
        self.md = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding='same'),
            nn.SELU(),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding='same'),
            nn.SELU(),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding='same'),
            nn.SELU(),
            nn.MaxPool2d((1, 5)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, 5, padding=(2, 0)),
            nn.SELU()
        )

    def forward(self, cfp):
        feat = self.md(cfp)
        return feat.squeeze(-1).permute(0, 2, 1)


class MelodyFeatureModule(nn.Module):
    def __init__(self):
        super(MelodyFeatureModule, self).__init__()
        self.embed = nn.Embedding(401, 32)

    def forward(self, x):
        return self.embed(x)


class ChartGenModel(nn.Module):
    def __init__(self):
        super(ChartGenModel, self).__init__()
        # hyper parameter
        self.block_length = 144
        self.query_shape = (1, 12)
        self.memory_flange = (4, 36)
        self.img_size = (500, 12)
        self.block_raster_scan = False
        self.alpha_temp = 0.45
        self.group_temp = 0.2
        self.color_temp = 0.15

        self.alpha_embed = nn.Sequential(
            nn.Linear(1, 32)
        )
        self.group_embed = nn.Embedding(4, 32)
        self.color_embed = nn.Embedding(6, 32)

        self.level_embed = nn.Embedding(40, 32)

        self.melody_module = AudioFeatureModule()

        self.melody_dropout = nn.Dropout(0.1)
        self.melody_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 8, batch_first=True), 2)
        self.chart_dropout = nn.Dropout(0.1)
        self.chart_decoder = ImageTransformerDecoder(
            ImageTransformerDecoder1dLayer(32, 8, block_length=self.block_length), 4)
        self.alpha_post = nn.Linear(32, 2)
        self.group_post = nn.Linear(32, 4)
        self.color_post = nn.Linear(32, 6)

    def forward(self, melody, chart, difficulty, predict=False):
        """
        chart channel is ordered as: alpha, group, color
        :param melody: CFP with shape [Batch, length, frq] or melody index with shape [Batch, length]
        :param chart: chart with shape [Batch, channel, length, width] or [Batch, 1, current_length, 1]
        :param difficulty: difficulty label with shape [Batch]
        :param predict: whether is in predict mode or not
        """

        melody = self.melody_module(melody)
        melody = add_timing_signal(melody)
        melody = self.melody_dropout(melody)
        encoded_melody = self.melody_encoder(melody)

        difficulty = self.level_embed(difficulty)

        if predict:
            chart = self.predict_fix_chart_shape(chart, self.block_raster_scan, self.query_shape, self.img_size)

        chart_alpha, chart_group, chart_color = chart.split(1, 1)
        chart_alpha = chart_alpha.squeeze(1).float()
        chart_group = chart_group.squeeze(1).int()
        chart_color = chart_color.squeeze(1).int()
        chart_alpha = self.alpha_embed(chart_alpha.unsqueeze(-1))
        chart_group = self.group_embed(chart_group)
        chart_color = self.color_embed(chart_color)
        chart = self.concat_charts(chart_alpha, chart_group, chart_color)
        chart_shape = chart.shape
        chart_frames, cols = chart_shape[1], chart_shape[2]
        # chart = right_shift_blockwise(chart, self.query_shape)
        chart = chart.reshape(chart_shape[0], chart_frames * cols, chart_shape[3])
        chart = right_shift_3d(chart)
        chart = chart.reshape(chart_shape)
        chart = add_timing_signal(chart)

        # put difficulty into chart data
        chart[:] += difficulty

        output = self.chart_decoder(chart, encoded_melody)
        out_alpha, out_group, out_color = self.unconcat_charts(output)
        out_alpha = self.alpha_post(out_alpha)
        out_group = self.group_post(out_group)
        out_color = self.color_post(out_color)

        if predict:
            out_alpha = self._infer_tempered_softmax_rand(out_alpha, self.alpha_temp)
            out_group = self._infer_tempered_softmax_rand(out_group, self.group_temp)
            out_color = self._infer_tempered_softmax_rand(out_color, self.color_temp)
            output = self.interleave_charts(out_alpha, out_group, out_color)
            output = output.reshape(output.size(0), 1, -1, 1)
            return output
        else:
            return out_alpha, out_group, out_color

    @staticmethod
    def _infer_tempered_softmax_rand(x: Tensor, temperature=1.0):
        assert temperature >= 0
        if temperature == 0.0:
            return x.argmax(-1)
        x_shape = x.shape
        x = x / temperature
        x = F.softmax(x, -1)
        x = x.reshape(-1, x_shape[-1])
        x = x.multinomial(1)
        x = x.reshape(x_shape[:-1])
        return x

    @staticmethod
    def interleave_charts(alpha: Tensor, group: Tensor, color: Tensor):
        """
        flatten chart channels by interleaving channels
        Shape: [N, L, W, E] where N is batch, L is time, W is width, E is embedding dimension
        :param alpha:
        :param group:
        :param color:
        :return:
        """
        return torch.flatten(torch.stack([alpha, group, color], 3), 2, 3)

    @staticmethod
    def uninterleave_charts(chart: Tensor):
        return chart.reshape(chart.size(0), -1, 12, 3, *chart.shape[3:]).unbind(3)

    @staticmethod
    def concat_charts(alpha: Tensor, group: Tensor, color: Tensor):
        """
        flatten chart channels by concatenating channels
        Shape: [N, L, W, E] where N is batch, L is time, W is width, E is embedding dimension
        :param alpha:
        :param group:
        :param color:
        :return:
        """
        return torch.cat([alpha, group, color], 2)

    @staticmethod
    def unconcat_charts(charts: Tensor):
        return charts.reshape(charts.size(0), -1, 3, 12, *charts.shape[3:]).unbind(2)

    @staticmethod
    def predict_fix_chart_shape(chart, block_raster_scan, query_shape, img_size):
        chart_batches = chart.shape[0]
        current_predict_len = chart.shape[2]
        chart = chart.permute(
            [0, 2, 3, 1])  # trick: convert to [Batch, current_length, 1, 1] to mimic behavior in Img2Img Transformer

        if block_raster_scan:
            assert img_size[1] * 3 % query_shape[1] == 0
            assert img_size[0] % query_shape[0] == 0
            total_block_width = img_size[1] * 3

            block_padding_factor = total_block_width * query_shape[0]

            # Note: while tensorflow does padding from dim 1 to dim 4, PyTorch does it from dim 4 to dim 1
            chart = F.pad(chart, [0, 0, 0, 0, 0, -current_predict_len % block_padding_factor])
            num_blocks = total_block_width // query_shape[1]
            chart_blocks = chart.reshape([chart_batches, -1, num_blocks, query_shape[0], query_shape[1]])
            chart = chart_blocks.permute([0, 1, 3, 2, 4])
        else:
            padding_factor = 3 * img_size[1]
            if current_predict_len % padding_factor == 0:
                # since the model will shift right, last value would be shifted out. So we intentionally set
                # pad length as the factor such that it will open a new line
                pad_length = padding_factor
            else:
                pad_length = -current_predict_len % padding_factor
            chart = F.pad(chart, [0, 0, 0, 0, 0, pad_length])
        chart = chart.reshape([chart_batches, -1, img_size[1], 3])
        chart = chart.permute([0, 3, 1, 2])  # put channel back in PyTorch style
        return chart
