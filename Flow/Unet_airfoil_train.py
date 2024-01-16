from mindspore.dataset import MindDataset
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.loss import MTLWeightedLossCell
from mindspore import Tensor
from model_airfoil import *
from mindspore.ops import operations as P
from mindflow.utils.check_func import check_param_type
import math
from utils import *

print(ms.__version__)
ms.set_context(device_target="GPU")
# print(ms.get_context("device_target"),ms.get_context("device_id"),ms.get_context("max_device_memory"))

max_value_list= [4.6, 1.0, 0.18418664581293126]
min_value_list= [-2.0, -0.18382872836312403, -0.1839052340212462]




class WaveletTransformLoss_unet(nn.LossBase):

    def __init__(self, wave_level=2, regroup=False):
        check_param_type(param=wave_level, param_name="wave_level", data_type=int)
        check_param_type(param=regroup, param_name="regroup", data_type=bool)
        super(WaveletTransformLoss_unet, self).__init__()
        self.abs = P.Abs()
        self.wave_level = wave_level
        self.regroup = regroup
        self.print = P.Print()
        if self.regroup:
            self.mtl = MTLWeightedLossCell(num_losses=3)
        else:
            if self.wave_level == 1:
                self.mtl = MTLWeightedLossCell(num_losses=4)
            else:
                self.mtl = MTLWeightedLossCell(num_losses=self.wave_level + 1)

    def construct(self, logit, label):
        l1_loss = P.ReduceMean()(self.abs(logit - label))

        if self.wave_level == 1:
            _, x_hl, x_lh, x_hh = self.dwt_split(logit)
            _, y_hl, y_lh, y_hh = self.dwt_split(label)
            hl_loss = P.ReduceMean()(self.abs(x_hl - y_hl))
            lh_loss = P.ReduceMean()(self.abs(x_lh - y_lh))
            hh_loss = P.ReduceMean()(self.abs(x_hh - y_hh))
            l_total = self.mtl((l1_loss, hl_loss, lh_loss, hh_loss))
        else:
            wave_losses = []
            for _ in range(self.wave_level):
                _, x_hl, x_lh, x_hh = self.dwt_split(logit)
                _, y_hl, y_lh, y_hh = self.dwt_split(label)
                hl_loss = P.ReduceMean()(self.abs(x_hl - y_hl))
                lh_loss = P.ReduceMean()(self.abs(x_lh - y_lh))
                hh_loss = P.ReduceMean()(self.abs(x_hh - y_hh))
                wave_loss_cur = hl_loss + lh_loss + hh_loss
                wave_losses.append(wave_loss_cur)
            wave_losses.append(l1_loss)
            l_total = self.mtl(wave_losses)
        return l_total

    @staticmethod
    def _split_data(data, axis=1):
        data = P.Transpose()(data, (1, 2, 3, 0))
        data_shape = data.shape
        data_re = []
        if axis == 1:
            data_re = P.Reshape()(data, (data_shape[0],
                                         data_shape[1] // 2,
                                         2,
                                         data_shape[2],
                                         data_shape[3]))
            data_re = P.Transpose()(data_re, (0, 2, 1, 3, 4))
        if axis == 2:
            data_re = P.Reshape()(data, (data_shape[0],
                                         data_shape[1],
                                         data_shape[2] // 2,
                                         2,
                                         data_shape[3]))
            data_re = P.Transpose()(data_re, (0, 1, 3, 2, 4))

        split_op = P.Split(axis, 2)
        data_split = split_op(data_re)
        data_01 = P.Squeeze()(data_split[0])
        data_02 = P.Squeeze()(data_split[1])
        return data_01, data_02

    def dwt_split(self, data):
        x01, x02 = self._split_data(data, axis=1)
        x1, x3 = self._split_data(x01 / 2, axis=2)
        x2, x4 = self._split_data(x02 / 2, axis=2)
        x_ll = x1 + x2 + x3 + x4
        x_hl = -x1 - x2 + x3 + x4
        x_lh = -x1 + x2 - x3 + x4
        x_hh = x1 - x2 - x3 + x4
        return x_ll, x_hl, x_lh, x_hh



def process_fn( data):

    _, h, w = data.shape
    xex = np.linspace(0, 5, h)
    f4 = np.array([math.exp(-x0) for x0 in xex])
    f4 = np.repeat(f4[:, np.newaxis], w, axis=1)

    aoa = data[0:1, ...]
    x = data[1:2, ...]
    y = data[2:3, ...]
    f4_x = x * f4
    f4_y = y * f4
    data = np.vstack((aoa, f4_x, f4_y))
    eps = 1e-8

    for i in range(0, data.shape[0]):
        max_value, min_value = max_value_list[i], min_value_list[i]
        data[i, :, :] = (data[i, :, :] - min_value) / (max_value - min_value + eps)
    return data.astype('float32')

def process_fn2( labels):
    label_shape = labels.shape
    img_size = (192, 384)
    patch_size = 16
    output_dim = label_shape[-1] // (patch_size * patch_size)
    labels = np.reshape(labels, (
                                    img_size[0] // patch_size,
                                    img_size[1] // patch_size,
                                    patch_size,
                                    patch_size,
                                    output_dim))

    labels = np.transpose(labels, (0, 2, 1, 3, 4))
    labels = np.reshape(labels, (
                                    img_size[0],
                                    img_size[1],
                                    output_dim))

    labels = np.transpose(labels, (2, 0, 1))
    return labels.astype('float32')





def display_error(error_name, error, error_list):
    """display error"""
    print(f'mean {error_name} : {error}, max {error_name} : {max(error_list)},'
          f' average {error_name} : {np.mean(error_list)},'
          f' min {error_name} : {min(error_list)}, median {error_name} : {np.median(error_list)}'
          )

def calculate_mean_error(label, pred):
    """calculate mean l1 error"""
    l1_error = np.mean(np.abs(label - pred))
    l1_error_u = np.mean(np.abs(label[..., 0] - pred[..., 0]))
    l1_error_v = np.mean(np.abs(label[..., 1] - pred[..., 1]))
    l1_error_p = np.mean(np.abs(label[..., 2] - pred[..., 2]))
    cp_error = np.mean(np.abs(label[..., 2][0, 0, :] - pred[..., 2][0, 0, :]))
    return l1_error, l1_error_u, l1_error_v, l1_error_p, cp_error



def get_label_and_pred(data, model):
    """get abel and pred"""
    labels = data["labels"]
    pred = model(data['inputs'])
    return labels.asnumpy(), pred.asnumpy()

def calculate_max_error(label, pred):
    """calculate max l1 error"""
    l1_error = np.max(np.max(np.abs(label - pred), axis=1), axis=1)
    l1_error_avg = np.mean(l1_error, axis=1).tolist()
    l1_error_u = l1_error[:, 0].tolist()
    l1_error_v = l1_error[:, 1].tolist()
    l1_error_p = l1_error[:, 2].tolist()
    cp_error = np.max(np.abs(label[..., 2][:, 0, :] - pred[..., 2][:, 0, :]), axis=1).tolist()
    return l1_error_avg, l1_error_u, l1_error_v, l1_error_p, cp_error

def calculate_eval_error(dataset, model, save_error=False, post_dir=None):
    """calculate evaluation error"""
    print("================================Start Evaluation================================")
    length = dataset.get_dataset_size()
    l1_error, l1_error_u, l1_error_v, l1_error_p, l1_error_cp = 0.0, 0.0, 0.0, 0.0, 0.0
    l1_error_list, l1_error_u_list, l1_error_v_list, l1_error_p_list, l1_error_cp_list, l1_avg_list = \
        [], [], [], [], [], []
    for data in dataset.create_dict_iterator(output_numpy=False):
        label, pred = get_label_and_pred(data, model)
        l1_max_step, l1_max_u_step, l1_max_v_step, l1_max_p_step, cp_max_step = calculate_max_error(label, pred)

        l1_avg = np.mean(np.mean(np.mean(np.abs(label - pred), axis=1), axis=1), axis=1).tolist()
        l1_error_list.extend(l1_max_step)
        l1_error_u_list.extend(l1_max_u_step)
        l1_error_v_list.extend(l1_max_v_step)
        l1_error_p_list.extend(l1_max_p_step)
        l1_error_cp_list.extend(cp_max_step)
        l1_avg_list.extend(l1_avg)

        l1_error_step, l1_error_u_step, l1_error_v_step, l1_error_p_step, cp_error_step = \
            calculate_mean_error(label, pred)
        l1_error += l1_error_step
        l1_error_u += l1_error_u_step
        l1_error_v += l1_error_v_step
        l1_error_p += l1_error_p_step
        l1_error_cp += cp_error_step
    l1_error /= length
    l1_error_u /= length
    l1_error_v /= length
    l1_error_p /= length
    l1_error_cp /= length
    display_error('l1_error', l1_error, l1_error_list)
    display_error('u_error', l1_error_u, l1_error_u_list)
    display_error('v_error', l1_error_v, l1_error_v_list)
    display_error('p_error', l1_error_p, l1_error_p_list)
    display_error('cp_error', l1_error_cp, l1_error_cp_list)
    if save_error:
        save_dir = os.path.join(post_dir, "Unet_error")
        # check_file_path(save_dir)
        print(f"eval error save dir: {save_dir}")
        np.save(os.path.join(save_dir, 'l1_error_list'), l1_error_list)
        np.save(os.path.join(save_dir, 'l1_error_u_list'), l1_error_u_list)
        np.save(os.path.join(save_dir, 'l1_error_v_list'), l1_error_v_list)
        np.save(os.path.join(save_dir, 'l1_error_p_list'), l1_error_p_list)
        np.save(os.path.join(save_dir, 'l1_error_cp_list'), l1_error_cp_list)
        np.save(os.path.join(save_dir, 'l1_error_avg_list'), l1_avg_list)
    print("=================================End Evaluation=================================")

mindrecord_name = "./data/train_dataset.mind"
batch_size=16
dataset = MindDataset(dataset_files=mindrecord_name, shuffle=False)
dataset = dataset.project(["inputs", "labels"])
print("samples:", dataset.get_dataset_size())


dataset_eval = MindDataset(dataset_files="./data/test_dataset.mind", shuffle=False)
dataset_eval = dataset_eval.project(["inputs", "labels"])

dataset = dataset.shuffle(batch_size*4)
dataset_eval = dataset_eval.shuffle(batch_size*4)

dataset = dataset.map(operations=process_fn,
                                        input_columns=["inputs"])
dataset = dataset.map(operations=process_fn2,
                                      input_columns=["labels"])

dataset_eval = dataset_eval.map(operations=process_fn,
                                      input_columns=["inputs"])
dataset_eval = dataset_eval.map(operations=process_fn2,
                                      input_columns=["labels"])


dataset = dataset.batch(batch_size)
dataset_eval = dataset_eval.batch(batch_size)

model=UNet(3,3)

# param_dict = ms.load_checkpoint("./output/model_airfoil_unet_850.ckpt")
# param_not_load, _ = ms.load_param_into_net(model, param_dict)
# print(param_not_load)

epochs =10000
lr=0.002
steps_per_epoch = dataset.get_dataset_size()

wave_loss = WaveletTransformLoss_unet(wave_level=1)
L1_loss=nn.L1Loss()
MSE_loss=nn.MSELoss()
# prepare optimizer

loss_scaler = ms.amp.DynamicLossScaler(scale_value=2**10, scale_factor=2, scale_window=50)

lr = get_warmup_cosine_annealing_lr(lr_init=lr,
                                    last_epoch=epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    warmup_epochs=1)

optimizer = nn.Adam(model.trainable_params()+ wave_loss.trainable_params(), learning_rate=Tensor(lr))
# optimizer = nn.SGD(model.trainable_params(), 1e-2)


def forward_fn(data, label):
    output = model(data)
    loss = wave_loss(output, label)
    # loss = loss_scaler.scale(loss)
    return loss


grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

def train_step(data, label):
    loss, grads = grad_fn(data, label)
    optimizer(grads)
    return loss



eval_interval = 50
save_ckt_interval = 100
plot_interval=100
# train process
for epoch in range(1, 1+epochs):
    # train
    time_beg = time.time()
    model.set_train()
    loss_all=0
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):

        loss = train_step(data, label)
        loss_all+=loss


    loss_all=loss_all/steps_per_epoch
    print(f"epoch: {epoch} train loss: {loss_all} epoch time: {time.time() - time_beg:.2f}s")
    # eval
    model.set_train(False)
    if epoch % eval_interval == 0:
        calculate_eval_error(dataset_eval, model)
    if epoch % save_ckt_interval == 0:
        ckpt_name = './output/model_airfoil_unet2_{}.ckpt'.format(epoch)
        ms.save_checkpoint(model, ckpt_name)
        print(f'{ckpt_name} save success')
