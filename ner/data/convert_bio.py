import json

key_dict = {'JobTitle-B': 'B-JOB',\
           'JobTitle-I': 'I-JOB',\
            'Company-B': 'B-ORG',\
            'Company-I': 'I-ORG',\
            'Person-B': 'B-PER',\
            'Person-I': 'I-PER'
            }

def convert_single(input_json):
    res = {}
    datastr = input_json['data']
    for i in range(len(datastr)):
        res[i] = 'O'

    #replace with label json
    labels = input_json['label']
    for (start,end,type) in labels:
        start_type = type + '-B'
        i_type = type + '-I'
        res[int(start)] = key_dict[start_type]
        for j in range((int(start)+1),(int(end))):
            res[j] = key_dict[i_type]

    return res

def save_label(input_json,output_json,save_path):
    with open(save_path, "a", encoding='utf-8') as w:
        for i in range(len(input_json['data'])):

            str = input_json['data'][i]

            label = output_json[i]
            if (label == '。 ' + 'O'):
                w.write('\n')
            else:
                w.write(str+' '+label)
                w.write('\n')

        # print(list)

    return

def main(file_path):
    data = []
    with open(file_path, 'r', encoding='utf8') as inp:
        for line in inp.readlines():
            data.append(json.loads(line.split('\n')[0]))
    print(data[0])

    #process single file
    for i in range(len(data)):
        res = convert_single(data[i])
        x = save_label(data[i], res,"./train.txt")
    print ('save train file success!')

    for i in range(1):
        res = convert_single(data[i])
        x = save_label(data[i], res,"./test.txt")
    print ('save test file success!')

    for i in range(1):
        res = convert_single(data[i])
        x = save_label(data[i], res,"./dev.txt")
    print ('save dev file success!')

def _read_data(input_file):
    """Reads a BIO data."""
    with open(input_file) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            # if len(contends) == 0 and words[-1] == '。':
            if len(contends) == 0:
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
    print (lines[0])

    return lines

if __name__ == '__main__':
    FILE_PATH = '/Users/liujunyi/Desktop/spottag/summit-training/ner/11-txts-annotation.txt'
    main(FILE_PATH)