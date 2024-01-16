#export MS_DEV_ENABLE_FALLBACK=0
#export MS_DEV_ENABLE_FALLBACK=0
#export GLOG_v = 1
python train_mindspore.py hparams/sepformer-libri2mix_new.yaml --data_folder /mnt/nvme1/LibriMix/Libri2Mix
