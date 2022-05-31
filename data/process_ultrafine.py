
import json
import os

path = os.getcwd()

# input data dir
dataset_path = os.path.join(path, 'release/crowd')
# output data dir
processed_dataset_path = os.path.join(path, 'processed_data')
os.mkdir(processed_dataset_path)
#  type vocab file
type_vocab_file_path = os.path.join(path, 'release/ontology/types.txt')


# Load and process typing vocabulary
typing_vocab = []
with open(type_vocab_file_path) as fin:
    for lines in fin:
        lines = lines.split()[0]
        lines = ' '.join(lines.split('_'))
        typing_vocab.append(lines)

processed_vocab_file_path = os.path.join(processed_dataset_path,'types.txt')
with open(processed_vocab_file_path, 'w+') as fout:
    fout.write('\n'.join(typing_vocab))

# process test, dev, train
for file in ['test', 'dev', 'train']:
    in_file_name = os.path.join(dataset_path, f'{file}.json')
    out_file_name = os.path.join(processed_dataset_path, f'{file}_processed.json')
    # load and process data
    data_lst = []
    idx = 0
    with open(in_file_name) as fin:
        for lines in fin:
            raw_dat = json.loads(lines)
            processed_dat = {}

            entity = raw_dat['mention_span']
            left_tokens = raw_dat['left_context_token']
            right_tokens = raw_dat['right_context_token']
            raw_annotations = raw_dat['y_str']
            annotations = []
            for items in raw_annotations:
                cleaned_annotations = ' '.join(items.split('_'))
                if cleaned_annotations in typing_vocab:
                    annotations.append(cleaned_annotations)
            if not annotations:
                continue
            premise = ' '.join(left_tokens + [entity] + right_tokens)

            processed_dat['premise'] = premise
            processed_dat['entity'] = entity
            processed_dat['annotation'] = annotations
            processed_dat['id'] = f'{file}{idx:04n}'
            data_lst.append(processed_dat)
            idx += 1

    # save processed path
    with open(out_file_name, 'w+') as fout:
        fout.write('\n'.join([json.dumps(items) for items in data_lst]))
