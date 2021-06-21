import re
import os
import pickle
import tokenization
import numpy as np

def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars

max_seq_length = 128
vocab_file = './albert_config/vocab.txt'
print(max_seq_length)
print(vocab_file)

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

with open(os.path.join('albert_base_ner_checkpoints', 'label2id.pkl'), 'rb') as rf:  # TODO need use label2id.pkl in model.tar.gz
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}
print(label2id)
print(id2label)

# text = '因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。'
text = '美国的华莱士，我和他谈笑风生。'
tokens = ['[CLS]']
tokens.extend(seg_char(text)[:max_seq_length-2])
tokens.append('[SEP]')
print(len(tokens), tokens)

input_ids = tokenizer.convert_tokens_to_ids(tokens)
for i in range(max_seq_length-len(tokens)):
    input_ids.append(0)
input_mask = [1 if i<len(tokens) else 0 for i in range(max_seq_length)]
segment_ids = [0 for _ in range(max_seq_length)]
label_ids = [0 for _ in range(max_seq_length)]
# input_ids = np.reshape(np.array(input_ids), (1, max_seq_length)).tolist()
# input_mask = np.reshape(np.array(input_mask), (1, max_seq_length)).tolist()
# segment_ids = np.reshape(np.array(segment_ids), (1, max_seq_length)).tolist()
# label_ids = np.reshape(np.array(label_ids), (1, max_seq_length)).tolist()
print(input_ids)
print(input_mask)


import requests
import time

start = time.time()
resp = requests.post('http://localhost:8501/v1/models/albert_chinese_ner_model:predict', json={"inputs": {"input_ids": input_ids}})
end = time.time()
pro = resp.json()['outputs'][0]
print(f"pro:{pro}, time consuming:{int((end - start) * 1000)}ms")
