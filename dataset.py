import json
import random
from torch.utils.data import Dataset


class TypingDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = []

        with open(label_file, "r", encoding="utf-8") as fin:
            label_lst = []
            for lines in fin:
                lines = lines.split()[0]
                lines = ' '.join(lines.split('_'))
                label_lst.append(lines)
            self.label_lst = label_lst
            self.general_lst = label_lst[0:9]
            self.fine_lst = label_lst[9:130]
            self.ultrafine_lst = label_lst[130:]

        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        for line in lines:
            line = json.loads(line)

            premise = line['premise']
            entity = line['entity']
            # could truncate generated annotation
            annotation = line['annotation']
            idx = line['id']
            annotation_general = list(set(annotation).intersection(set(self.general_lst)))
            annotation_fine = list(set(annotation).intersection(set(self.fine_lst)))
            annotation_ultrafine = list(set(annotation).intersection(set(self.ultrafine_lst)))

            self.data.append([premise, entity, annotation, annotation_general, annotation_fine, annotation_ultrafine, idx])


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)