# ChartGAN - Rhythm Game Chart Generator using GAN

This project purposed a GAN model in generating rhythm game chart as a series of images.

## Prerequisite

This project requires PyTorch and numpy to work. In particular, this project is developed in the following environment:
```
torch == 1.11.0
numpy == 1.21.5
librosa == 0.9.1
```
Librosa is used only in preprocessing dataset. Librosa is strongly recommanded to be installed in `--user` mode in Windows environment or librosa caching during import might freeze the process due to no admin privilege.

## Prepare the dataset

To download and preprocess the datamined dataset, use the following commands:
```
python download_prsk_dataset.py
python preprocess_prsk_dataset.py
```
These commands will download the dataset into `prsk_dataset` folder, and preprocess music into tensors, saved as `*.npy` format.

## Train the model

To train the model, use the following command:
```
python gan_train.py
```
The following arguments can be inputted to control the behavior of the training function:

`--num_epochs EPOCHS`: number of epochs for each stage. Example: `--num_epochs 10`

`--data_dir DIR`: path of dataset. Example: `--data_dir prsk_dataset`

`--chkpt_dir DIR`: path where checkpoint will be saved and loaded. Example: `--chkpt_dir .`

`--chkpt_fmt`: filename format of checkpoint file. Example: `--chkpt_fmt "chartgen_{gd}_stage_{stage}.pth"`

`--batch_size BATCH`: batch size. Example: `--batch_size 4`

`--lr RATE`: learning rate. Example: `--lr 0.001`

`--end_stage STAGE`: Ending stage (exclusive). Example: `--end_stage 5`

`--start_stage STAGE`: Starting stage. Example: `--stage_stage 0`

`--use_cfp`: A flag to tell trainer to use CFP representation instead of log mel-scaled spectrogram

`--long_stride`: A flag to tell trainer to use 200 window length instead of 30 window length

`--seq_len LENGTH`: sequence length during training. Example: `--seq_len 5`

`--ce_weight WEIGHT`: weight of cross-entropy loss. Example: `--ce_weight 80`

## Generate a chart

To generate a chart, use the following command:
```
python gan_infer.py -m <checkpoint_path> -f <audio_path> -l <difficulty>
```
This will generate a chart, in numpy array, into the same of audio sample with the same name, but append with `_chart.npy`

This script also support the following, optional arguments:

`--use_cfp`: A flag to tell trainer to use CFP representation instead of log mel-scaled spectrogram

`--long_stride`: A flag to tell trainer to use 200 window length instead of 30 window length

## View a chart

To display a chart, use the following command:
```
python display_chart.py <chart file>
```
This will display a chart. This script supports this additional argument:

`-c` or `--note-type`: choose which channel to be displayed. For chart with only 1 channel, this argument is required. The following type is supported:

- `alpha` or `a`: Note alpha
- `group` or `g`: Note group
- `color` or `c`: Note color