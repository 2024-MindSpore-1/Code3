import mindspore as ms 
import mindspore.ops as ops
import mindspore.nn as nn
from tqdm import tqdm
from utils import update_lr, Meter, cal_score

grad_fn = None
optimizer = None

    
def init_grad_fn_and_optimizer(optim):
    global optimizer
    global grad_fn
    optimizer = optim
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def collate_fn_myself(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)
    zeros = ops.Zeros()
    images, image_masks = zeros((len(proper_items), channel, max_height, max_width),ms.float32), zeros((len(proper_items), 1, max_height, max_width),ms.float32)
    labels, label_masks = zeros((len(proper_items), max_length),ms.int32), zeros((len(proper_items), max_length),ms.float32)
    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images_numpy = images.asnumpy()
        proper_items_numpy = proper_items[i][0].asnumpy()
        images_numpy[i][:, :h, :w] = proper_items_numpy
        images = ms.Tensor(images_numpy)
        proper_items[i] = list(proper_items[i])
        proper_items[i][0] = ms.Tensor(proper_items_numpy)
        image_masks_numpy = image_masks.asnumpy()
        image_masks_numpy[i][:, :h, :w] = 1
        image_masks = ms.Tensor(image_masks_numpy)
        l = proper_items[i][1].shape[0]
        labels_numpy = labels.asnumpy()
        proper_items_numpy = proper_items[i][1].asnumpy()
        labels_numpy[i][:l] = proper_items_numpy
        labels = ms.Tensor(labels_numpy)
        proper_items[i][1] = ms.Tensor(proper_items_numpy)
        label_masks_numpy = label_masks.asnumpy()
        label_masks_numpy[i][:l] = 1
        label_masks = ms.Tensor(label_masks_numpy)
    return images, image_masks, labels, label_masks

def train(params, model, optimizer, epoch, train_loader):
    def forward_fn(images, image_masks, labels, label_masks):
        probs, counting_preds, word_loss, counting_loss = model(images, image_masks, labels, label_masks)
        return word_loss, counting_loss, probs
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    def train_step(images, image_masks, labels, label_masks):
        (word_loss, counting_loss, probs), grads = grad_fn(images, image_masks, labels, label_masks)
        loss = word_loss + counting_loss
        grads = ops.clip_by_global_norm(grads, 100)
        loss = ops.depend(loss, optimizer(grads))
        return loss, word_loss, counting_loss,probs
    model.set_train()
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0
    batch_idx = 0
    train_loader_iter = train_loader.create_dict_iterator()
    with tqdm(train_loader.create_tuple_iterator(), total=train_loader.get_dataset_size()//params['train_parts']) as pbar:
        while(True):
            batch_idx += 1
            batch_size = params['batch_size']
            batch_datas = []
            for batch_data in train_loader_iter:
                batch_datas.append((batch_data['data'],batch_data['label']))
                batch_size-=1
                if batch_size == 0:
                    break
            images, image_masks, labels, label_masks = collate_fn_myself(batch_datas)
            batch, time = labels.shape[:2]
            loss, word_loss, counting_loss, probs = train_step(images, image_masks, labels, label_masks)
            loss_meter.add(float(loss))
            wordRate, ExpRate = cal_score(probs, labels, label_masks)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            pbar.set_description(f'{epoch+1} word_loss:{float(word_loss):.4f} counting_loss:{float(counting_loss):.4f} WRate:{float(word_right) / float(length):.4f} '
                                     f'ERate:{float(exp_right) / float(cal_num):.4f}')
            if batch_idx >= train_loader.get_dataset_size() //params['batch_size']:
                break
            pbar.update(params['batch_size'])
            break
    return loss_meter.mean, word_right / length, exp_right / cal_num


def eval(params, model, epoch, eval_loader, writer=None):
    model.set_train(False)
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0
    batch_idx = 0
    eval_loader_iter = eval_loader.create_dict_iterator()
    with tqdm(eval_loader.create_tuple_iterator(), total=eval_loader.get_dataset_size()//params['batch_size']) as pbar:
        while(True):
            batch_idx += 1
            batch_size = params['batch_size']
            batch_datas = []
            for batch_data in eval_loader_iter:
                batch_datas.append((batch_data['data'],batch_data['label']))
                batch_size-=1
                if batch_size == 0:
                    break
            images, image_masks, labels, label_masks = collate_fn_myself(batch_datas)
            batch, time = labels.shape[:2]
            probs, counting_preds, word_loss, counting_loss = model(images, image_masks, labels, label_masks, is_train=False)
            loss = word_loss + counting_loss
            loss_meter.add(float(loss))

            wordRate, ExpRate = cal_score(probs, labels, label_masks)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            pbar.set_description(f'{epoch+1} word_loss:{float(word_loss):.4f} counting_loss:{float(counting_loss):.4f} WRate:{float(word_right) / float(length):.4f} '
                                     f'ERate:{float(exp_right) / float(cal_num):.4f}')
            if batch_idx >= eval_loader.get_dataset_size() //params['batch_size']:
                break
        return loss_meter.mean, word_right / length, exp_right / cal_num
        