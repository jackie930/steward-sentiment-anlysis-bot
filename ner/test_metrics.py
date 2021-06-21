import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def metric_fn(label_ids, predictions):
    cm = confusion_matrix(label_ids, predictions)
    print(cm)
    precision = precision_score(label_ids, predictions, average="macro")
    recall = recall_score(label_ids, predictions, average="macro")
    f = f1_score(label_ids, predictions, average="macro")
    return {
        "test_precision": precision,
        "test_recall": recall,
        "test_f": f,
    }


test_filename = 'data/test.txt'
label_test_filename = 'albert_base_ner_checkpoints/label_test.txt'
token_test_filename = 'albert_base_ner_checkpoints/token_test.txt'

with open('albert_base_ner_checkpoints/label2id.pkl', 'rb') as rf:  # TODO need use label2id.pkl in model.tar.gz
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}
print(label2id)
print(id2label)

fin1 = open(test_filename, 'r')
fin2 = open(label_test_filename, 'r')
fin3 = open(token_test_filename, 'r')

labels = []
predictions = []
label_ids = []
prediction_ids = []
tokens = []
while True:
    pred = fin2.readline().strip()
    token = fin3.readline().strip()
    if not pred:
        break
    if pred == '[CLS]':
        continue
    line = fin1.readline().strip()
    if not line or line == '':
        continue
    label = line.split(' ')[1]
    labels.append(label)
    predictions.append(pred)
    label_ids.append(label2id[label])
    prediction_ids.append(label2id[pred])
    tokens.append(token)

fin1.close()
fin2.close()
fin3.close()

result = metric_fn(label_ids, prediction_ids)
print(result)


entity_label2id = {}
entity_id2label = {}
for key, value in label2id.items():
    if '-' in key:
        new_key = key.split('-')[1]
    else:
        new_key = key
    if new_key not in entity_label2id:
        entity_label2id[new_key] = len(entity_label2id)
entity_id2label = {value: key for key, value in entity_label2id.items()}
print(entity_label2id)
print(entity_id2label)

entity_labels = []
entity_predictions = []
entity_label_ids = []
entity_prediction_ids = []
for i in range(len(labels)):
    if labels[i] == 'O' and predictions == 'O':
        continue
    if labels[i].startswith('B-'):
        # print('case1:', i, labels[i], predictions[i], tokens[i])
        entity_label = labels[i].split('-')[1]
        
        if '-' in predictions[i]:
            entity_pred = predictions[i].split('-')[1]
        else:
            entity_pred = predictions[i]
            
        for j in range(i+1, len(labels)):
            if labels[j] == 'O':
                break
            if labels[j] != predictions[j]:  # TODO 一旦有一个识别错误，则认为预测错误（记为X）
                entity_pred = 'X'
                j += 1
                break
        
        #print('case1:', i, j, labels[i:j], predictions[i:j], tokens[i:j], entity_label, entity_pred)
        entity_labels.append(entity_label)
        entity_predictions.append(entity_pred)
        entity_label_ids.append(entity_label2id[entity_label])
        entity_prediction_ids.append(entity_label2id[entity_pred])
    elif predictions[i].startswith('B-'):
        entity_pred = predictions[i].split('-')[1]
        
        if '-' in labels[i]:
            entity_label = labels[i].split('-')[1]
        else:
            entity_label = labels[i]
            
        for j in range(i+1, len(labels)):
            if predictions[j] == 'O':
                break
            if labels[j] != predictions[j]:  # TODO 一旦有一个识别错误，则认为预测错误（记为X）
                entity_label = 'X'
                j += 1
                break
            
        #print('case2:', i, j, labels[i:j], predictions[i:j], tokens[i:j], entity_label, entity_pred)
        entity_labels.append(entity_label)
        entity_predictions.append(entity_pred)
        entity_label_ids.append(entity_label2id[entity_label])
        entity_prediction_ids.append(entity_label2id[entity_pred])

    
entity_result = metric_fn(entity_label_ids, entity_prediction_ids)
print(entity_result)