import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Union, Tuple


class TemperedSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperedSoftmax, self).__init__()
        assert temperature > 0
        self.temperature = temperature

    def forward(self, x, dim=-1):
        return F.softmax(x / self.temperature, dim=dim)



class CrossEntropyGanLoss(nn.Module):
    """
    Implementation of Adversarial Loss, following pix2pix's implementation as:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, *, real_label=1.0, fake_label=0.0):
        super(CrossEntropyGanLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target(self, prediction, target_is_real):
        tgt_tensor: Tensor
        if target_is_real:
            tgt_tensor = self.real_label
        else:
            tgt_tensor = self.fake_label
        return tgt_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        return self.loss(prediction, self.get_target(prediction, target_is_real))


class LossSensitiveGanLoss(nn.Module):
    """
    implementation of Loss Sensitive GAN (loss part) with inspiration from
    https://github.com/maple-research-lab/glsgan-gp/blob/master/lsgan-gp.py

    set negative_slope to 0 will get LSGAN, and set to 1 will get WGAN
    any negative_slope that larger than 1 will be rejected, as it violate:
    C(a) >= a for any a in Real domain

    Related paper: https://arxiv.org/abs/1701.06264
    """

    def __init__(self, negative_slope, distance_lambda=0.01):
        super(LossSensitiveGanLoss, self).__init__()
        assert negative_slope <= 1
        self.loss_regulator = nn.LeakyReLU(negative_slope)
        self.distance = nn.PairwiseDistance(1)
        self.distance_lambda = distance_lambda

    def forward(self, real_sample, fake_sample, d_output_real, d_output_fake):
        """
        Calculate loss of GLSGAN. In the paper, margin of real space and fake space is calculated by real sample and
        generated sample only regardless conditioned or not, so we are safe to use here.

        :param real_sample: Real sample. Here we want real charts. [N, 10, 200, 12]
        :param fake_sample: Fake sample. Here we want generated charts that can paired to real_sample. Same shape with
        real sample
        :param d_output_real: Output of discriminator with real sample input. [N]
        :param d_output_fake: Output of discriminator with generated sample input. [N]
        :return: scalar loss
        """
        real_sample = real_sample.reshape(real_sample.size(0), -1)
        fake_sample = fake_sample.reshape_as(real_sample)
        dist = self.distance(real_sample, fake_sample)
        dist = dist.mul(self.distance_lambda)
        loss = self.loss_regulator(d_output_real - d_output_fake + dist)
        return loss.mean()


class BatchNormConvTranspose2d(nn.Module):
    def __init__(self, in_feature: int, out_feature: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias=False, activation: Union[nn.Module, None] = nn.ReLU()):
        super(BatchNormConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_feature, out_feature, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_feature)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class BatchNormConv2d(nn.Module):
    def __init__(self, in_feature: int, out_feature: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias=False, activation: Union[nn.Module, None] = nn.LeakyReLU(0.2)):
        super(BatchNormConv2d, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_feature)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class RealChartFixer(nn.Module):
    """
    Encode a real chart to match the output of ChartGanGen
    """
    def __init__(self):
        super(RealChartFixer, self).__init__()

    def forward(self, x: Tensor):
        if x.dim() == 5:
            # sequential charts. [L, N, C, H, W]
            group = x[:, :, 1, :, :] if x.size(2) > 2 else x[:, :, 0, :, :]
            color = x[:, :, 2, :, :] if x.size(2) > 2 else x[:, :, 1, :, :]
            group = F.one_hot(group.long(), 4).float()
            color = F.one_hot(color.long(), 6).float()
            chart = torch.cat([group, color], dim=-1).permute(0, 1, 4, 2, 3)
            return chart
        else:
            assert x.dim() == 4
            # single chart. [N, C, H, W]
            group = x[:, 1, :, :] if x.size(1) > 2 else x[:, 0, :, :]
            color = x[:, 2, :, :] if x.size(1) > 2 else x[:, 1, :, :]
            group = F.one_hot(group, 4)
            color = F.one_hot(color, 6)
            chart = torch.cat([group, color], dim=-1).permute(0, 3, 1, 2)
            return chart


class GenChartFixer(nn.Module):
    """
    temper-softmax the output of ChartGanGen so that it looks like one-hot argmax outcome
    """
    def __init__(self):
        super(GenChartFixer, self).__init__()
        self.softmaxer = TemperedSoftmax(1e-3)

    def forward(self, x: Tensor):
        if x.dim() == 5:
            # sequential charts. [L, N, C, H, W]
            group = x[:, :, :4, :, :]
            color = x[:, :, 4:, :, :]
            group = self.softmaxer(group, 2)
            color = self.softmaxer(color, 2)
            chart = torch.cat([group, color], dim=2)
            return chart
        else:
            assert x.dim() == 4
            # single chart. [N, C, H, W]
            group = x[:, :4, :, :]
            color = x[:, 4:, :, :]
            group = self.softmaxer(group, 1)
            color = self.softmaxer(color, 1)
            chart = torch.cat([group, color], dim=1)
            return chart


class ChartMaker(nn.Module):
    """
    from output of ChartGanGen, retrieve a chart's group and color information
    """
    def __init__(self):
        super(ChartMaker, self).__init__()

    def forward(self, x: Tensor):
        if x.dim() == 5:
            # sequential charts. [L, N, C, H, W]
            group = x[:, :, :4, :, :]
            color = x[:, :, 4:, :, :]
            group = group.argmax(dim=2)
            color = color.argmax(dim=2)
            return group, color
        elif x.dim() == 4:
            # single chart. [N, C, H, W]
            group = x[:, :4, :, :]
            color = x[:, 4:, :, :]
            group = group.argmax(dim=1)
            color = color.argmax(dim=1)
            return group, color
        else:
            assert x.dim() == 3
            # single chart, unbatched. [C, H, W]
            group = x[:4, :, :]
            color = x[4:, :, :]
            group = group.argmax(dim=0)
            color = color.argmax(dim=0)
            return group, color


class CfpFeatureSubmodule(nn.Module):
    def __init__(self):
        super(CfpFeatureSubmodule, self).__init__()
        self.md = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding='same'),
            nn.SELU(),
            nn.MaxPool2d((4, 1)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding='same'),
            nn.SELU(),
            nn.MaxPool2d((4, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding='same'),
            nn.SELU(),
            nn.MaxPool2d((5, 1)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 5, padding=(0, 2)),
            nn.SELU()
        )

    def forward(self, cfp):
        assert cfp.dim() == 4
        feat = self.md(cfp)
        return feat


class MelFeatureSubmodule(nn.Module):
    def __init__(self):
        super(MelFeatureSubmodule, self).__init__()
        self.conv1 = BatchNormConv2d(3, 10, (3, 7), 1, 0, activation=nn.ReLU())
        self.pool1 = nn.MaxPool2d((3, 1))
        self.conv2 = BatchNormConv2d(10, 20, 3, 1, 0, activation=nn.ReLU())
        self.pool2 = nn.MaxPool2d((3, 1))
        self.gru = nn.GRU(160, 200, num_layers=2, batch_first=True)

    def forward(self, lmel80: Tensor):
        assert lmel80.dim() == 4  # [N, C, F, T]. we don't handle outer sequence here
        x = self.conv1(lmel80)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # if we input [N, 3, 80, 15], we should have [N, 20, 8, 7] here
        x = x.flatten(1, 2)  # [N, 160, 7]
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        return x[:,-1,:]  #[N, 200]



def create_level_embedder(embed_dim: int):
    return nn.Embedding(40, embed_dim)


class ChartGanDiss(nn.Module):
    def __init__(self, level_embedder: nn.Embedding = None,
                 audio_feature_module: Union[CfpFeatureSubmodule, MelFeatureSubmodule] = None,
                 *, window_length=200, square_conv=True):
        super(ChartGanDiss, self).__init__()
        if level_embedder is None:
            level_embedder = create_level_embedder(32)
        if audio_feature_module is None:
            audio_feature_module = CfpFeatureSubmodule()

        self.level_embed_dim = level_embedder.embedding_dim
        self._level_embed_base = level_embedder
        self.level_embedder = level_embedder

        self._cfp_base = audio_feature_module
        self.cfp_feature_extractor = audio_feature_module
        if square_conv:
            assert window_length == 30
            self.downsamplers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(10, 32, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                ),  # 16x16
                BatchNormConv2d(32, 64, 4, 2, 1),  # 8x8
                BatchNormConv2d(64, 128, 4, 2, 1),  # 4x4
            ])
            self.image_to_feature = nn.Conv2d(128, 256, 4, 1, 0, bias=False)
        else:
            assert window_length == 200
            self.downsamplers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(10, 16, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                ),  # 128x8
                BatchNormConv2d(16, 32, 4, 2, (1, 5)),  # 64x8
                BatchNormConv2d(32, 64, 4, 2, (1, 5)),  # 32x8
                BatchNormConv2d(64, 128, 4, 2, 1),  # 16x4
                BatchNormConv2d(128, 256, 4, 2, (1, 3)),  # 8x4
                BatchNormConv2d(256, 512, 4, 2, (1, 3)),  # 4x4
            ])
            self.image_to_feature = nn.Conv2d(512, 256, 4, 1, 0, bias=False)
        if isinstance(audio_feature_module, CfpFeatureSubmodule):
            fc_wanted_size = self.level_embed_dim + window_length + 256
        else:
            assert isinstance(audio_feature_module, MelFeatureSubmodule)
            fc_wanted_size = self.level_embed_dim + 200 + 256
        self.fc = nn.Sequential(
            nn.Linear(fc_wanted_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def lock(self):
        """
        freeze unshared parameters.
        """
        # import copy
        # self.level_embedder = copy.deepcopy(self._level_embed_base).requires_grad_(False)
        # self.cfp_feature_extractor = copy.deepcopy(self._cfp_base).requires_grad_(False)

        for m in self.downsamplers:
            for param in m.parameters():
                param.requires_grad = False
        for param in self.image_to_feature.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False

    def unlock(self):
        """
        unfreeze unshared parameters
        """
        # self.level_embedder = self._level_embed_base
        # self.cfp_feature_extractor = self._cfp_base

        for m in self.downsamplers:
            for param in m.parameters():
                param.requires_grad = True
        for param in self.image_to_feature.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, chart, level, audio_feat):
        """

        :param chart: Chart. [N, 10, 256, 16]/[L,N,10,256,16]. In channel, first 4 is group onehot, last 6 is color onehot
        :param level: Level. [N] / [L,N]
        :param audio_feat: CFP representation. [N, 3, 400, 200]/[L,N,3,400,200]; or 
        lmel80 (stack of stft, log mel-scaled, 80 bins). [N, 3, 80, 30] / [L, N, 3, 80, 30]
        :return:
        """
        seq_input = False
        seq_len = None
        batch_size = None
        if chart.dim() == 5 and level.dim() == 2 and audio_feat.dim() == 5:
            # time sequence input
            seq_input = True
            seq_len = level.size(0)
            batch_size = level.size(1)
            # TODO: support recurrent processing
            chart = chart.reshape(seq_len * batch_size, *chart.shape[2:])
            level = level.reshape(seq_len * batch_size)
            audio_feat = audio_feat.reshape(seq_len * batch_size, *audio_feat.shape[2:])
        elif chart.dim() == 4 and level.dim() == 1 and audio_feat.dim() == 4:
            # single time input
            batch_size = level.size(0)
        else:
            raise ValueError(f"mismatch dim size: {chart.dim()}, {level.dim()}, {audio_feat.dim()}")
        level = self.level_embedder(level)
        audio_feat = self.cfp_feature_extractor(audio_feat)
        audio_feat = audio_feat.reshape(audio_feat.size(0), audio_feat.size(-1))  # squeeze channel and bins

        img = chart
        for downsampler in self.downsamplers:
            img = downsampler(img)

        img_feature = self.image_to_feature(img)
        img_feature = img_feature.reshape(img_feature.shape[:2])  # [N, C]
        features = torch.cat([level, audio_feat, img_feature], dim=-1)
        output: Tensor
        output = self.fc(features)
        if seq_input:
            output = output.reshape(seq_len, batch_size, 1)
        return output


class ChartGanGen(nn.Module):
    def __init__(self, z_dim=64, level_embedder: nn.Embedding = None,
                 audio_feature_module: Union[CfpFeatureSubmodule, MelFeatureSubmodule] = None, *, window_length=200, square_conv=True):
        super(ChartGanGen, self).__init__()
        if level_embedder is None:
            level_embedder = create_level_embedder(32)
        if audio_feature_module is None:
            audio_feature_module = CfpFeatureSubmodule()

        self.z_dim = z_dim
        self.level_embed_dim = level_embedder.embedding_dim
        self.level_embedder = level_embedder
        if isinstance(audio_feature_module, CfpFeatureSubmodule):
            self.gru_wanted_input_size = window_length + self.z_dim + self.level_embed_dim
        else:
            assert isinstance(audio_feature_module, MelFeatureSubmodule)
            self.gru_wanted_input_size = 200 + self.z_dim + self.level_embed_dim


        self.cfp_feature_extractor = audio_feature_module
        self.gru = nn.GRU(input_size=self.gru_wanted_input_size, hidden_size=256)

        if square_conv:
            assert window_length == 30
            self.feature_to_image = BatchNormConvTranspose2d(256, 128, 4, 1, 0)
            self.upsamplers = nn.ModuleList([
                BatchNormConvTranspose2d(128, 64, 4, 2, 1),  # 8x8
                BatchNormConvTranspose2d(64, 32, 4, 2, 1),  # 16x16
                nn.Sequential(
                    nn.ConvTranspose2d(32, 10, 4, 2, 1, bias=False),  # 32x32
                    nn.ReLU()
                )
            ])
        else:
            assert window_length == 200
            self.feature_to_image = BatchNormConvTranspose2d(256, 512, 4, 1, 0)
            self.upsamplers = nn.ModuleList([
                BatchNormConvTranspose2d(512, 256, 4, 2, (1, 3)),  # 8x4
                BatchNormConvTranspose2d(256, 128, 4, 2, (1, 3)),  # 16x4
                BatchNormConvTranspose2d(128, 64, 4, 2, 1),  # 32x8
                BatchNormConvTranspose2d(64, 32, 4, 2, (1, 5)),  # 64x8
                BatchNormConvTranspose2d(32, 16, 4, 2, (1, 5)),  # 128x8
                nn.Sequential(
                    nn.ConvTranspose2d(16, 10, 4, 2, 1, bias=False),  # 256x16
                    nn.ReLU()
                )
            ])


    def forward(self, z: Tensor, level: Tensor, audio_feat: Tensor, *, initial_h: Tensor = None):
        """

        :param z: Noise tenser. [N, z_dim] / [L, N, z_dim]
        :param level: level tensor. [N] / [L, N]
        :param audio_feat: CFP representation of audio. [N, 3, 400, 200] / [L, N, 3, 400, 200]; or 
        lmel80 (stack of stft, log mel-scaled, 80 bins). [N, 3, 80, 30] / [L, N, 3, 80, 30]
        :param initial_h: initial hidden state for recurrent. [L, N, 256]
        :return: generated image [N, 10, 256, 16] / [L, N, 10, 256, 16]; and hidden state of GRU [L, N, 256]
        """
        assert z.size(-1) == self.z_dim
        seq_input = False
        seq_len = None
        batch_size = None
        if z.dim() == 3 and level.dim() == 2 and audio_feat.dim() == 5:
            # time sequence input
            seq_input = True
            seq_len = level.size(0)
            batch_size = level.size(1)
        elif z.dim() == 2 and level.dim() == 1 and audio_feat.dim() == 4:
            # single time input
            batch_size = level.size(0)
        else:
            raise ValueError(f"mismatch dim size: {z.dim()}, {level.dim()}, {audio_feat.dim()}")
        level = self.level_embedder(level)
        cfp_before_shape = None
        if seq_input:
            audio_feat = audio_feat.reshape(seq_len * batch_size, *audio_feat.shape[2:])
        audio_feat = self.cfp_feature_extractor(audio_feat)
        if seq_input:
            audio_feat = audio_feat.reshape(seq_len, batch_size, audio_feat.size(-1))
        else:
            audio_feat = audio_feat.reshape(batch_size, audio_feat.size(-1))  # squeeze channel and bins
        cat_features = torch.cat([level, audio_feat, z], dim=-1)
        if not seq_input:
            cat_features = cat_features.unsqueeze(0)

        img_prefeatures, h = self.gru(cat_features, initial_h)
        if seq_input:
            img_prefeatures = img_prefeatures.reshape(seq_len * batch_size, img_prefeatures.size(-1), 1, 1)
        else:
            img_prefeatures = img_prefeatures.reshape(batch_size, img_prefeatures.size(-1), 1, 1)
        img_4x4 = self.feature_to_image(img_prefeatures)
        img = img_4x4
        for upsampler in self.upsamplers:
            img = upsampler(img)
        if seq_input:
            img = img.reshape(seq_len, batch_size, *img.shape[1:])

        return img, h
