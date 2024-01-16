import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, json, datetime
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()




def load_lora_config(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"]
    )
    return get_peft_model(model, config)

PROMPT_PATTERN = "问：{}"
SEP_PATTERN = "\n答： "
def create_prompt(question):
    return PROMPT_PATTERN.format(question), SEP_PATTERN


def create_prompt_ids(tokenizer, question, max_src_length):
    prompt, sep = create_prompt(question)
    sep_ids = tokenizer.encode(
        sep, 
        add_special_tokens = True
    )
    sep_len = len(sep_ids)
    special_tokens_num = 2
    prompt_ids = tokenizer.encode(
        prompt, 
        max_length = max_src_length - (sep_len - special_tokens_num),
        truncation = True,
        add_special_tokens = False
    )

    return prompt_ids + sep_ids


def create_inputs_and_labels(tokenizer, question, answer, device):
    prompt = create_prompt_ids(tokenizer, question, max_src_length)
    completion = tokenizer.encode(
        answer, 
        max_length = max_dst_length,
        truncation = True,
        add_special_tokens = False
    )

    inputs = prompt + completion + [eop]
    labels = [-100] * len(prompt) + completion + [eop] 
    
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return inputs, labels

def get_attention_mask(tokenizer, input_ids, device):
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)
    attention_mask = torch.ones((seq_len, seq_len), device=device)
    attention_mask.tril_()
    attention_mask[..., :context_len] = 1
    attention_mask.unsqueeze_(0)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask


def get_position_ids(tokenizer, input_ids, device, position_encoding_2d=True):
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)

    mask_token = mask if mask in seq else gmask
    use_gmask = False if mask in seq else gmask

    mask_position = seq.index(mask_token)

    if position_encoding_2d:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position
        block_position_ids = torch.cat((
            torch.zeros(context_len, dtype=torch.long, device=device),
            torch.arange(seq_len - context_len, dtype=torch.long, device=device) + 1
        ))
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position
    
    return position_ids

class QADataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
 

    def __getitem__(self, index):
        item_data = self.data[index]
        tokenizer = self.tokenizer
        input_ids, labels = create_inputs_and_labels(
            tokenizer, 
            device=device,
            **item_data
        )
        
        attention_mask = get_attention_mask(tokenizer, input_ids, device)
        position_ids = get_position_ids(tokenizer, input_ids, device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    position_ids = []
    
    for obj in batch:
        input_ids.append(obj['input_ids'])
        labels.append(obj['labels'])
        attention_mask.append(obj['attention_mask'])
        position_ids.append(obj['position_ids'])
        
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask), 
        'labels': torch.stack(labels),
        'position_ids':torch.stack(position_ids)
    }

class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss



checkpoint = "/data1/ckw/kewei_models/chatglm-6b-096"
revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"
model = AutoModel.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)

model = load_lora_config(model)
model.load_state_dict(torch.load(f"./output_new/chatglm-6b-lora.pt"), strict=False)

model.half().cuda().eval()

app = FastAPI()


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
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)