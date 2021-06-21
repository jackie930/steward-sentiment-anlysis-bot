import json
import tokenization
import sagemaker
from sagemaker.predictor import json_serializer, json_deserializer

max_seq_length = 128
vocab_file = './albert_config/vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

id2label = json.load(open('./data/id2label.json', 'r'))
print('id2label:', id2label)

sagemaker_session = sagemaker.Session()
# predictor = sagemaker.predictor.Predictor('tensorflow-training-2021-05-25-04-08-14-571', sagemaker_session, json_serializer, json_deserializer)
predictor = sagemaker.predictor.Predictor('albert-chinese-ner-2021-05-25-05-16-47-002', sagemaker_session, json_serializer, json_deserializer)

def lambda_handler(event, context):
    query = event['queryStringParameters']
    text = query['text']
    
    # text = '因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。'
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(text)[:max_seq_length-2])
    tokens.append('[SEP]')

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    for i in range(max_seq_length-len(tokens)):
        input_ids.append(0)
        
    data = {'instances': [{'input_ids': [input_ids]}]}
    
    result = predictor.predict(data)
    
    result_label = [id2label[str(id)] for id in result['predictions'][0] if id != 0]
    
    nes = {}
    ne = ''
    nep = ''
    for i in range(len(result_label)):
        if result_label[i].startswith('B-'):
            ne = tokens[i]
            nep = result_label[i].replace('B-', '')
        elif result_label[i].startswith('I-'):
            ne += tokens[i]
        else:
            if ne != '':
                nes[ne] = nep
                ne = ''
                nep = ''
    
    return {
        'statusCode': 200,
        'body': json.dumps(nes, ensure_ascii=False)
    }
