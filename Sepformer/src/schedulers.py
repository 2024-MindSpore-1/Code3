import mindspore
import mindspore.nn as nn

from ..utils import checkpoints

@checkpoints.register_checkpoint_hooks
class ReduceLROnPlateau:
    """Learning rate scheduler which decreases the learning rate if the loss
    function of interest gets stuck on a plateau, or starts to increase.
    The difference from NewBobLRScheduler is that, this one keeps a memory of
    the last step where do not observe improvement, and compares against that
    particular loss value as opposed to the most recent loss.

    Arguments
    ---------
    lr_min : float
        The minimum allowable learning rate.
    factor : float
        Factor with which to reduce the learning rate.
    patience : int
        How many epochs to wait before reducing the learning rate.

    Example
    -------
    # >>> from torch.optim import Adam
    # >>> from speechbrain.nnet.linear import Linear
    # >>> inp_tensor = torch.rand([1,660,3])
    # >>> model = Linear(n_neurons=10, input_size=3)
    # >>> optim = Adam(lr=1.0, params=model.parameters())
    # >>> output = model(inp_tensor)
    # >>> scheduler = ReduceLROnPlateau(0.25, 0.5, 2, 1)
    # >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    # >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=11.0)
    # >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=13.0)
    # >>> curr_lr,next_lr=scheduler([optim],current_epoch=4, current_loss=14.0)
    # >>> next_lr
    0.5
    """

    def __init__(self, lr_min=1e-8, factor=0.5, patience=2, dont_halve_until_epoch=65):
        self.lr_min = lr_min
        self.factor = factor
        self.patience = patience
        self.patience_counter = 0
        self.losses = []
        self.dont_halve_until_epoch = dont_halve_until_epoch
        self.anchor = 99999

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        next_lr : float
            The learning rate after the update.
        """
        # for opt in optim_list:
        #     current_lr = opt.param_groups[0]["lr"]

        for opt in optim_list:
            current_lr = opt.get_lr()[0].asnumpy()

            if current_epoch <= self.dont_halve_until_epoch:
                next_lr = current_lr
                self.anchor = current_loss
            else:
                if current_loss <= self.anchor:
                    self.patience_counter = 0
                    next_lr = current_lr
                    self.anchor = current_loss
                elif (
                    current_loss > self.anchor
                    and self.patience_counter < self.patience
                ):
                    self.patience_counter = self.patience_counter + 1
                    next_lr = current_lr
                else:
                    next_lr = current_lr * self.factor
                    self.patience_counter = 0

            # impose the lower bound
            next_lr = max(next_lr, self.lr_min)

        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = [{
            "losses": self.losses,
            "anchor": self.anchor,
            "patience_counter": self.patience_counter,
        }]
        mindspore.save_checkpoint(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device  # Not used
        data = mindspore.load_checkpoint(path)
        self.losses = data["losses"]
        self.anchor = data["anchor"]
        self.patience_counter = data["patience_counter"]