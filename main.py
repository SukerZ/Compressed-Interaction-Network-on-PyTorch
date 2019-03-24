import pdb
from data import *
from torch.utils.data import *
from xDeepFM import *

if __name__ == '__main__':
    file = {"train": "data/train.userid.txt", "test": "data/test.userid.txt"}
    field_num = 33
    d, l = feature_dict(file, field_num)
    dataset = libffm_data(file, field_num, d)
    batch_num = 256
    dataloader = DataLoader(dataset, batch_size = batch_num, shuffle = True)

    model = xDeepFM(field_num, l)
    for epoch in range(10):
        for i, data in enumerate(dataloader):
            features, labels = data
            model.train(features, labels )

        print("Epoch", epoch + 1, "已完成。");

        loss = 0
        for i, data in enumerate(dataloader):
            features, labels = data
            loss += model.test(features, labels )

        print("Loss =", loss.item() ); model.save()