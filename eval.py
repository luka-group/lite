import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import random
import json
from heapq import nlargest
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import AutoTokenizer
from dataset import TypingDataset
from model import roberta_mnli_typing
import argparse

def eval(args, eval_dataset, model, tokenizer):
    curr_time = time.strftime("%H_%M_%S_%b_%d_%Y", time.localtime())
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1, collate_fn=lambda x: zip(*x))
    type_vocab = eval_dataset.label_lst

    eval_res = []
    for sample in tqdm(eval_dataloader, desc='eval progress'):
        premise, entity, annotation, _, _, _, idx = [items for items in sample]
        premise = str(premise[0])
        entity = str(entity[0])
        annotation = list(annotation[0])
        idx = str(idx[0])
        res = {'id': idx, 'premise': premise, 'entity': entity, 'annotation': annotation}

        res_buffer = {}
        for batch_id in range(0, len(type_vocab), args.batch):
            dat_buffer = type_vocab[batch_id: batch_id+args.batch]
            sequence = [f'{premise}{2*tokenizer.sep_token}{entity} is a {label}.' for label in dat_buffer]
            inputs = tokenizer(sequence, padding=True, return_tensors='pt').to(args.device)
            outputs = model(**inputs)[:, -1]
            confidence = outputs.detach().cpu().numpy().tolist()
            for idx in range(len(dat_buffer)):
                res_buffer[dat_buffer[idx]] = confidence[idx]

        confidence_ranking = {labels: res_buffer[labels] for labels in res_buffer
                             if res_buffer[labels] > args.threshold}
        confidence_ranking = {k: v for k, v in sorted(confidence_ranking.items(), key=lambda item: -item[1])}
        res['confidence_ranking'] = confidence_ranking

        eval_res.append(res)

    return eval_res


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir',
                        type=str, 
                        default='',
                        help='fine-tuned model dir path')
    parser.add_argument('--eval_data_path', 
                        type=str, 
                        default='/data/processed_data/dev_processed.json',
                        help='dev/test file path')
    parser.add_argument('--type_vocab_file', 
                        type=str, 
                        default='', 
                        help='type vocab file path')
    parser.add_argument('--batch', 
                        type=int, 
                        default=8,
                        help='To batchify candidate type words or phrases')
    parser.add_argument('--threshold', 
                        type=float, 
                        default=0.0, 
                        help='Threshold for confident score, 0 to print the full ranking of candidates')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise ValueError("Cannot find model checkpoint: {}".format(args.model_dir))

    try:
        # output file would be modelFileName_evalFileName.json
        output_suffix = args.eval_data_path.split('/')[-1]
        output_path = os.path.join(args.model_dir, f'Evaluation_{output_suffix}')
    except:
        raise ValueError("Cannot generate output file name, please manually input")

    model = roberta_mnli_typing()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

    chkpt_path = os.path.join(args.model_dir, 'model')
    chkpt = torch.load(chkpt_path, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    print(f'Evaluating {args.model_dir}\n on {args.eval_data_path} '
          f'\n result file will be saved to {output_path}')
    eval_dataset = TypingDataset(args.eval_data_path, args.type_vocab_file)
    eval_res = eval(args, eval_dataset, model, tokenizer)

    # save res file
    with open(output_path, 'w+') as fout:
        fout.write("\n".join([json.dumps(items) for items in eval_res]))
    
if __name__ == "__main__":
    main()