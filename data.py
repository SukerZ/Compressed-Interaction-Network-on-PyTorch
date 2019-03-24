import pdb
from torch.utils.data import *

def id(word):
    tokens = word.strip().split(':')
    field_id = int(tokens[0]) - 1
    feature_id = tokens[1]
    value = float(tokens[2])

    return field_id, feature_id, value

def feature_dict(files, field_num):
    d = {}
    for i in range(field_num):
        d[i] = {}

    for k in files.keys():
        with open(files[k] ) as rd:
            f = rd.readlines()
            for line in f:
                tmp = line.strip().split('%')
                line = tmp[0]
                cols = line.split(' ')
                for word in cols[1:]:
                    if not word.strip():
                        continue
                    field_id, feature_id, _ = id(word)
                    if feature_id not in d[field_id].keys():
                        d[field_id][feature_id] = len(d[field_id])

    l = []
    for i in range(field_num):
        l.append(len(d[i] ) )

    return d, l

class libffm_data(Dataset):
    def __init__(self, file, field_num, d):
        self.file = file; self.field_num = field_num
        self.labels = []; self.features_i = []; self.features_v = []
        self.features = []
        with open(file["train"], 'r') as rd:
            f = rd.readlines()
            for line in f:
                tmp = line.strip().split('%')
                line = tmp[0]
                cols = line.strip().split(' ')
                label = float(cols[0].strip() )

                if label > 1e-6:
                    label = 1
                else:
                    label = 0

                feat_i = []
                for i in range(self.field_num ):
                    feat_i.append([])

                feat_v = []
                for i in range(self.field_num ):
                    feat_v.append([])

                for word in cols[1:]:
                    if not word.strip():
                        continue
                    field_id, feature_id, feature_value = id(word)
                    feat_i[field_id].append(d[field_id][feature_id] )
                    feat_v[field_id].append(feature_value)

                self.features.append((feat_i, feat_v) )
                self.labels.append(label)

    def __getitem__(self, idx):
        label = self.labels[idx];
        return self.features[idx], label

    def __len__(self):
        return len(self.labels)