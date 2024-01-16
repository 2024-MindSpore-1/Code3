import mindspore as ms
from mindspore import ops
def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    # 将输入的句子进行编码，返回的input_ids类型为tensor
    input_ids = ms.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    # 模型输入input_ids和标签input_ids，输出模型预测结果logits和损失loss，outputs为一个tuple
    outputs = model(input_ids, labels=input_ids)
    # 取出tuple中的第一项（即损失loss）和第二项（即预测结果logits）
    loss, logits = outputs[:2]
    # 返回exp(loss)
    return ops.exp(loss)