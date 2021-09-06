import numpy as np
from torch.utils.data.dataset import Dataset



def classify_label(dataset, num_classes: int):
    list_label2indices = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list_label2indices[datum[1]].append(idx)
    return list_label2indices


def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1

        print(f'{client}: {nums_data}')


def partition_train_teach(list_label2indices: list, num_data_train: int, seed=None):
    random_state = np.random.RandomState(seed)
    list_label2indices_train = []
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_train.append(indices[:num_data_train // 10])
        list_label2indices_teach.append(indices[num_data_train // 10:])
    return list_label2indices_train, list_label2indices_teach


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)
