#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Test ChatGLM"""
import random
import unittest
import numpy as np

import mindspore
from mindspore import Tensor

from mindnlp.models.glm.chatglm import ChatGLMForConditionalGeneration
from mindnlp.transforms.tokenizers import ChatGLMTokenizer


# In[2]:


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)

    # mindspore RNGs
    mindspore.set_seed(seed)

    # numpy RNG
    np.random.seed(seed)


def ids_tensor(shape, vocab_size):
    """Creates a random int32 tensor of the shape within the vocab size"""
    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(random.randint(0, vocab_size - 1))

    return mindspore.Tensor(values, dtype=mindspore.int64).view(shape)


def get_model_and_tokenizer():
    """get model and tokenizer"""
    model = ChatGLMForConditionalGeneration.from_pretrained("chatglm-6b")

    tokenizer = ChatGLMTokenizer.from_pretrained("chatglm-6b")
    return model, tokenizer


# In[3]:


model = ChatGLMForConditionalGeneration.from_pretrained("chatglm-6b")


# In[4]:


# while True:
#     try:
#         model = ChatGLMForConditionalGeneration.from_pretrained("chatglm-6b")
#         break
#     except:
#         continue





# In[5]:


tokenizer = ChatGLMTokenizer.from_pretrained("chatglm-6b")



# In[14]:


from mindnlp.models.glm.chatglm import LogitsProcessorList,InvalidScoreLogitsProcessor


# In[17]:


def stream_chat(tokenizer, query: str, history= None, max_length: int = 2048,
                do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    """stream chat"""
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                  "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer(prompt)
    for outputs in model.stream_generate(**inputs, **gen_kwargs):
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = model.process_response(response)
        new_history = history + [(query, response)]
        yield response, new_history


# In[32]:


import mindspore as ms





# In[39]:


from fastapi import FastAPI, Request


# In[ ]:


app = FastAPI()





# In[50]:


import uvicorn, json, datetime


# In[51]:

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    set_random_seed(42)
    inputs = tokenizer(prompt)
    inputs = Tensor([inputs])
    outputs = model.generate(
        inputs,
        do_sample=do_sample,
        max_length=max_length,
        num_beams=num_beams,
        jit=True,
        use_bucket=use_bucket,
        bucket_num=bucket_num
    )

    outputs = outputs.asnumpy().tolist()[0]
    out_sentence = tokenizer.decode(outputs, skip_special_tokens=True)
    
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": out_sentence,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(out_sentence) + '"'
    print(log)
    return answer


# In[52]:


# # 调用函数示例
# request_data = {
#     "prompt": "你好",
#     "history": [],
#     "max_length": 1024,
#     "top_p": 0.7,
#     "temperature": 0.95
# }

# response = await create_item(request_data)
# print(response)


# In[ ]:


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

