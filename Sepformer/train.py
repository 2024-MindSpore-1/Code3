import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import stop_gradient

import src.core as core

# Define training procedure
class Separation(core.Brain):
    # def __init__(self):

    # def speech_distortions(self, mix, targets, noise=None):
    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        expand_dims = ops.ExpandDims()
        concat = ops.Concat(axis=-1)
        stack = ops.Stack()

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        # mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        # targets = torch.cat(
        #     [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
        #     dim=-1,
        # ).to(self.device)
        targets = concat([expand_dims(targets[i][0], -1) for i in range(self.hparams.num_spks)])

        # Add speech distortions
        if stage == core.Stage.TRAIN:
            # with torch.no_grad():
            # use_speedperturb: True
            # use_rand_shift: False
            if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                mix, targets = self.add_speed_perturb(targets, mix_lens)

                mix = targets.sum(-1)

                # use_wham_noise: False
                if self.hparams.use_wham_noise:
                    # noise = noise.to(self.device)
                    len_noise = noise.shape[1]
                    len_mix = mix.shape[1]
                    min_len = min(len_noise, len_mix)

                    # add the noise
                    mix = mix[:, :min_len] + noise[:, :min_len]

                    # fix the length of targets also
                    targets = targets[:, :min_len, :]
            targets = stop_gradient(targets)
            mix = stop_gradient(mix)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        # mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        mix_w = stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        # est_source = torch.cat(
        #     [
        #         self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
        #         for i in range(self.hparams.num_spks)
        #     ],
        #     dim=-1,
        # )
        est_source = concat(
            [
                expand_dims(self.hparams.Decoder(sep_h[i]), -1)
                for i in range(self.hparams.num_spks)
            ],
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.shape[1]
        T_est = est_source.shape[1]
        if T_origin > T_est:
            # est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
            pad = nn.Pad(paddings=((0, 0), (0, T_origin - T_est), (0, 0)))
            est_source = pad(est_source)
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        return self.hparams.loss(targets, predictions)

    # fit_batch 实现困难
    def fit_batch(self, batch):
        """Trains one batch"""

        size = ops.Size()

        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.use_wham_noise:
            noise = batch.noise_sig[0]
        else:
            noise = None

        # num_spks: 2
        # if self.hparams.num_spks == 3:
        #     targets.append(batch.s3_sig)

        # auto_mix_prec: True # Set it to True for mixed precision
        # note torch.tensor = True 未实现，准备用mindspore自带的混合精度实现，core中self.scaler，因为缺少torch.cuda.amp.GradScaler()
        # if self.auto_mix_prec:
        #     self.scaler = torch.cuda.amp.GradScaler()
        if self.auto_mix_prec:
            # raise ValueError("Automatic mixed precision not implemented")
            # with autocast():
            predictions, targets = self.compute_forward(
                mixture, targets, core.Stage.TRAIN, noise
            )
            loss = self.compute_objectives(predictions, targets)

            # hard threshold the easy dataitems
            # threshold_byloss: True
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if size(loss_to_keep) > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            # loss_upper_lim: 999999
            if (
                    loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()

                # clip_grad_norm: 5
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = mindspore.Tensor(0)
        else:
            predictions, targets = self.compute_forward(
                mixture, targets, core.Stage.TRAIN, noise
            )
            loss = self.compute_objectives(predictions, targets)

            # threshold_byloss: True
            # threshold: -30
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if size(loss_to_keep) > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            # loss_upper_lim: 999999
            if (
                    loss < self.hparams.loss_upper_lim and size(loss) > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                # loss.data = torch.tensor(0).to(self.device)
                loss.data = mindspore.Tensor(0)
        self.optimizer.zero_grad()

        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]

        # with torch.no_grad():
        predictions, targets = self.compute_forward(mixture, targets, stage)
        loss = self.compute_objectives(predictions, targets)

        # return loss.detach()
        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == core.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == core.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                    self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # self.checkpointer.save_and_keep_only(
            #     meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            # )
        elif stage == core.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        # use_speedperturb: True
        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            # use_rand_shift: False
            # if self.hparams.use_rand_shift:
            #     # Performing random_shift (independently on each source)
            #     recombine = True
            #     for i in range(targets.shape[-1]):
            #         rand_shift = torch.randint(
            #             self.hparams.min_shift, self.hparams.max_shift, (1,)
            #         )
            #         new_targets[i] = new_targets[i].to(self.device)
            #         new_targets[i] = torch.roll(
            #             new_targets[i], shifts=(rand_shift[0],), dims=1
            #         )

            # Re-combination
            if recombine:
                # use_speedperturb: True
                if self.hparams.use_speedperturb:
                    # targets = torch.zeros(
                    #     targets.shape[0],
                    #     min_len,
                    #     targets.shape[-1],
                    #     device=targets.device,
                    #     dtype=torch.float,
                    # )
                    zeros = ops.Zeros()
                    targets = zeros(
                        (targets.shape[0], min_len, targets.shape[-1]),
                        mindspore.float32
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    # if self.hparams.num_spks == 3:
                    #     targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, core.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    # mixture_signal = torch.stack(
                    #     [mixture] * self.hparams.num_spks, dim=-1
                    # )
                    stack = ops.Stack(axis=-1)
                    mixture_signal = stack([mixture] * self.hparams.num_spks)

                    # mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].T.asnumpy(),
                        predictions[0].T.asnumpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].T.asnumpy(),
                        mixture_signal[0].T.asnumpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

