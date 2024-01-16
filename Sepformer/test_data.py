import logging
import sys
from create_directory import create_experiment_directory
from hyperpyyaml import load_hyperpyyaml
from dataprocess import dataio, dataset, data_pipeline
from myparser import parse_arguments
from mindspore.dataset import GeneratorDataset, RandomSampler


if __name__ == "__main__":
    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
         hparams = load_hyperpyyaml(fin, overrides)
    # Logger info
    logger = logging.getLogger(__name__)

    create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Data preparation
    from prepare_data import prepare_librimix

    prepare_librimix(
        hparams["data_folder"],
        hparams["save_folder"],
        hparams["num_spks"],
        hparams["skip_prep"],
        hparams["use_wham_noise"],
        hparams["sample_rate"]
    )
    # Create dataset objects
    train_data, valid_data, test_data, label = dataset.dataio_prep(hparams)

    #Create dataloader
    train_dataset = GeneratorDataset(train_data, label)
    sample = RandomSampler()
    train_dataset.add_sampler(sample)
    train_dataset = train_dataset.batch(1, True)
    iterator = train_dataset.create_tuple_iterator()
    
    
    # this is for test:
    cnt = 0
    for item in iterator:
      print(item[0])
      print(" ")
      cnt = cnt + 1
      if cnt > 10:
        break
      
    