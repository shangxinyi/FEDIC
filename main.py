from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import classify_label, partition_train_teach, show_clients_data_distribution, Indices2Dataset, \
    label_indices2indices
from Dataset.sampling_dirichlet import clients_indices
import numpy as np
from torch import stack, div, max, eq, no_grad, tensor, add, mul, ones, zeros, ones_like, nn, sigmoid, cat
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn.functional import softmax, log_softmax
from torch.utils.data.dataloader import DataLoader
from Model.model_feature import ResNet_cifar_feature
from tqdm import tqdm
import torch.nn.functional as F
import copy
import torch
import random



class Ensemble_highway(nn.Module):
    def __init__(self):
        super(Ensemble_highway, self).__init__()
        # calibration
        self.ensemble_scale = nn.Parameter(ones(10, 1))
        self.ensemble_bias = nn.Parameter(zeros(1))

        self.logit_scale = nn.Parameter(ones(10))
        self.logit_bias = nn.Parameter(zeros(10))
        self.classifier2 = nn.Linear(in_features=256, out_features=1)
        self.carry_values = []
        self.weight_values = []
    def forward(self, step, clients_feature, clients_logit, new_logit):
        all_logits_weight = torch.mm(clients_logit[0], self.ensemble_scale)
        all_logits_weight = all_logits_weight + self.ensemble_bias
        all_logits_weight_sigmoid = sigmoid(all_logits_weight)
        for one_logit in clients_logit[1:]:
            new_value = torch.mm(one_logit, self.ensemble_scale)
            new_value = new_value + self.ensemble_bias
            new_value_sigmoid = sigmoid(new_value)
            all_logits_weight_sigmoid = cat((all_logits_weight_sigmoid, new_value_sigmoid), dim=1)
        norm1 = all_logits_weight_sigmoid.norm(1, dim=1)
        norm1 = norm1.unsqueeze(1).expand_as(all_logits_weight_sigmoid)
        all_logits_weight_norm = all_logits_weight_sigmoid / norm1
        all_logits_weight_norm = all_logits_weight_norm.t()
        weighted_logits = sum([
            one_weight.view(-1, 1) * one_logit
            for one_logit, one_weight in zip(clients_logit, all_logits_weight_norm)
        ]
        )
        avg_weight = ([1.0 / 8] * 8)
        weighted_feature = sum(
            [
                one_weight * one_feature
                for one_feature, one_weight in zip(clients_feature, avg_weight)
            ]
        )
        calibration_logit = weighted_logits * self.logit_scale + self.logit_bias
        carry_gate = self.classifier2(weighted_feature)
        carry_gate_sigmoid = sigmoid(carry_gate)
        finally_logit = carry_gate_sigmoid * calibration_logit + (1 - carry_gate_sigmoid) * new_logit
        return finally_logit


class Global(object):
    def __init__(self,
                 dataset_global_teaching,
                 num_classes: int,
                 total_steps: int,
                 mini_batch_size: int,
                 mini_batch_size_unlabled: int,
                 lr_global_teaching: float,
                 temperature: float,
                 device: str,
                 seed,
                 server_steps,
                 num_online_clients,
                 ld: float,
                 ensemble_ld: float,
                 unlabeled_data):
        self.model = ResNet_cifar_feature(resnet_size=8, scaling=4,
                                          save_activations=False, group_norm_num_groups=None,
                                          freeze_bn=False, freeze_bn_affine=False, num_classes=num_classes)
        self.model1 = ResNet_cifar_feature(resnet_size=8, scaling=4,
                                          save_activations=False, group_norm_num_groups=None,
                                          freeze_bn=False, freeze_bn_affine=False, num_classes=num_classes)
        self.model2 = ResNet_cifar_feature(resnet_size=8, scaling=4,
                                          save_activations=False, group_norm_num_groups=None,
                                          freeze_bn=False, freeze_bn_affine=False, num_classes=num_classes)
        self.model.to(device)
        self.model1.to(device)
        self.model2.to(device)
        self.highway_model = Ensemble_highway()
        self.highway_model.to(device)
        self.dict_global_params = self.model.state_dict()
        self.dataset_global_teaching = dataset_global_teaching
        self.total_steps = total_steps
        self.mini_batch_size = mini_batch_size
        self.mini_batch_size_unlabled = mini_batch_size_unlabled
        self.ce_loss = CrossEntropyLoss()
        self.lr_global_teaching = lr_global_teaching
        self.optimizer = Adam(self.model.parameters(), lr=lr_global_teaching, weight_decay=0.0002)
        self.highway_optimizer = Adam(self.highway_model.parameters(), lr=lr_global_teaching)
        self.fedavg_optimizer = Adam(self.model2.parameters(), lr=lr_global_teaching, weight_decay=0.0002)
        self.temperature = temperature
        self.device = device
        self.epoch_acc = []
        self.epoch_acc_eval = []
        self.epoch_loss = []
        self.epoch_avg_ensemble_acc = []
        self.init_fedavg_acc = []
        self.init_ensemble_acc = []
        self.disalign_ensemble_acc = []
        self.disalign_ensemble_eval_acc = []
        self.epoch_acc_fed_min = []
        self.epoch_acc_fed_max = []
        self.random_state = np.random.RandomState(seed)
        self.server_steps = server_steps
        self.num_online_clients = num_online_clients
        self.ld = ld
        self.ensemble_ld = ensemble_ld
        self.num_classes = num_classes
        self.unlabeled_data = unlabeled_data

    def update_distillation_highway_feature(self, round, list_dicts_local_params: list,
                                            list_nums_local_data: list, data_global_test, batch_size_test):
        self._initialize_for_model_fusion(copy.deepcopy(list_dicts_local_params), list_nums_local_data)
        self.model2.load_state_dict(self.dict_global_params)
        self.model2.train()
        for hard_step in tqdm(range(100)):
            total_indices = [i for i in range(len(self.dataset_global_teaching))]
            batch_indices = self.random_state.choice(total_indices, self.mini_batch_size, replace=False)
            images = []
            labels = []
            for idx in batch_indices:
                image, label = self.dataset_global_teaching[idx]
                images.append(image)
                label = tensor(label)
                labels.append(label)
            images = stack(images, dim=0)
            images = images.to(self.device)
            labels = stack(labels, dim=0)
            labels = labels.to(self.device)
            _, fedavg_outputs = self.model2(images)
            fedavg_hard_loss = self.ce_loss(fedavg_outputs, labels)
            self.fedavg_optimizer.zero_grad()
            fedavg_hard_loss.backward()
            self.fedavg_optimizer.step()
        self.model2.eval()
        self.highway_model.train()
        for ensemble_step in tqdm(range(100)):
            total_indices = [i for i in range(len(self.dataset_global_teaching))]
            batch_indices = self.random_state.choice(total_indices, self.mini_batch_size, replace=False)
            images = []
            labels = []
            for idx in batch_indices:
                image, label = self.dataset_global_teaching[idx]
                images.append(image)
                label = tensor(label)
                labels.append(label)
            images = stack(images, dim=0)
            images = images.to(self.device)
            labels = stack(labels, dim=0)
            labels = labels.to(self.device)

            ensemble_feature_temp, ensemble_logit_temp = self.features_logits(images, copy.deepcopy(
                list_dicts_local_params))
            _, fedavg_new_logits = self.model2(images)
            ensemble_avg_logit_finally = self.highway_model(ensemble_step, ensemble_feature_temp,
                                                            ensemble_logit_temp, fedavg_new_logits)
            ensemble_hard_loss = self.ce_loss(ensemble_avg_logit_finally, labels)
            self.highway_optimizer.zero_grad()
            ensemble_hard_loss.backward()
            self.highway_optimizer.step()
        self.highway_model.eval()
        self.model.load_state_dict(self.dict_global_params)
        self.model.train()
        for step in tqdm(range(100)):
            total_indices_unlabeled = [i for i in range(len(self.unlabeled_data))]
            batch_indices_unlabeled = self.random_state.choice(total_indices_unlabeled, self.mini_batch_size_unlabled, replace=False)
            images_unlabeled = []
            for idx in batch_indices_unlabeled:
                image, _ = self.unlabeled_data[idx]
                images_unlabeled.append(image)
            images_unlabeled = stack(images_unlabeled, dim=0)
            images_unlabeled = images_unlabeled.to(self.device)
            total_indices_labeled = [j for j in range(len(self.dataset_global_teaching))]
            batch_indices_labeled = self.random_state.choice(total_indices_labeled, self.mini_batch_size, replace=False)
            images_labeled = []
            labels_train = []
            for idx in batch_indices_labeled:
                image, label = self.dataset_global_teaching[idx]
                images_labeled.append(image)
                label = tensor(label)
                labels_train.append(label)
            images_labeled = stack(images_labeled, dim=0)
            images_labeled = images_labeled.to(self.device)
            labels_train = stack(labels_train, dim=0)
            labels_train = labels_train.to(self.device)
            teacher_feature_temp, teacher_logits_temp = self.features_logits(images_unlabeled,
                                                                             copy.deepcopy(
                                                                                 list_dicts_local_params))
            _, fedavg_unlabeled_logits = self.model2(images_unlabeled)
            logits_teacher = self.highway_model(round, teacher_feature_temp, teacher_logits_temp,
                                                fedavg_unlabeled_logits)
            _, logits_student = self.model(images_unlabeled)
            x = log_softmax(logits_student / self.temperature, dim=1)
            y = softmax(logits_teacher / self.temperature, dim=1)
            soft_loss = F.kl_div(x, y.detach(), reduction='batchmean')
            _, logits_student_train = self.model(images_labeled)
            hard_loss = self.ce_loss(logits_student_train, labels_train)
            total_loss = add(mul(soft_loss, self.ld), mul(hard_loss, 1 - self.ld))
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        self.dict_global_params = self.model.state_dict()
        self.model.eval()
        acc = self.eval(data_global_test, batch_size_test)
        self.epoch_acc.append(acc)
        print(f'Distillation_test')
        print(acc)

    def features_logits(self, images, list_dicts_local_params):
        list_features = []
        list_logits = []
        for dict_local_params in list_dicts_local_params:
            self.model1.load_state_dict(dict_local_params)
            self.model1.eval()
            with no_grad():
                local_feature, local_logits = self.model1(images)
                list_features.append(copy.deepcopy(local_feature))
                list_logits.append(copy.deepcopy(local_logits))
        return list_features, list_logits

    def _initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        for name_param in tqdm(self.dict_global_params):
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            self.dict_global_params[name_param] = value_global_param

    def eval(self, data_test, batch_size_test: int):
        self.model.load_state_dict(self.dict_global_params)
        self.model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0

            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return copy.deepcopy(self.dict_global_params)


class Local(object):
    def __init__(self,
                 global_params,
                 data_client,
                 num_classes: int,
                 num_epochs_local_training: int,
                 batch_size_local_training: int,
                 lr_local_training: float,
                 device: str):
        self.model = ResNet_cifar_feature(resnet_size=8, scaling=4,
                                          save_activations=False, group_norm_num_groups=None,
                                          freeze_bn=False, freeze_bn_affine=False, num_classes=num_classes)
        self.model.to(device)
        self.model.load_state_dict(global_params)
        self.data_client = data_client
        self.num_epochs = num_epochs_local_training
        self.batch_size = batch_size_local_training
        self.ce_loss = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr_local_training)
        self.device = device
        self.lr = lr_local_training

    def train(self, round):
        self.model.train()
        for epoch in range(self.num_epochs):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.model(images)
                loss = self.ce_loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def upload_params(self):
        return copy.deepcopy(self.model.state_dict())


def disalign():
    args = args_parser()
    random_state = np.random.RandomState(args.seed)
    # Load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_train)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)
    unlabeled_data = datasets.CIFAR100(args.path_cifar100, transform=transform_train)

    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    list_label2indices_train, list_label2indices_teach = partition_train_teach(list_label2indices, args.num_data_train,
                                                                               args.seed)
    list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices_train), args.num_classes,
                                                   args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    show_clients_data_distribution(data_local_training, list_client2indices, args.num_classes)

    indices2data_teach = Indices2Dataset(data_local_training)
    indices2data_teach.load(label_indices2indices(list_label2indices_teach))

    global_model = Global(dataset_global_teaching=indices2data_teach,
                          num_classes=args.num_classes,
                          total_steps=args.total_steps,
                          mini_batch_size=args.mini_batch_size,
                          mini_batch_size_unlabled=args.mini_batch_size_unlabeled,
                          lr_global_teaching=args.lr_global_teaching,
                          temperature=args.temperature,
                          device=args.device,
                          seed=args.seed,
                          server_steps=args.server_steps,
                          num_online_clients=args.num_online_clients,
                          ld=args.ld,
                          ensemble_ld=args.ensemble_ld,
                          unlabeled_data=unlabeled_data)
    total_clients = list(range(args.num_clients))

    for r in range(1, args.num_rounds + 1):
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        print(online_clients)
        list_dicts_local_params = []
        list_nums_local_data = []
        # local training
        for client in tqdm(online_clients, desc='local training'):
            indices2data = Indices2Dataset(data_local_training)
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(global_params=copy.deepcopy(dict_global_params),
                                data_client=data_client,
                                num_classes=args.num_classes,
                                num_epochs_local_training=args.num_epochs_local_training,
                                batch_size_local_training=args.batch_size_local_training,
                                lr_local_training=args.lr_local_training,
                                device=args.device)
            local_model.train(round=r)
            dict_local_params = copy.deepcopy(local_model.upload_params())
            list_dicts_local_params.append(dict_local_params)
        # global update
        print(f'Round: [{r}/{args.num_rounds}] Global Updating')
        global_model.update_distillation_highway_feature(r, copy.deepcopy(list_dicts_local_params),
                                                         list_nums_local_data, data_global_test, args.batch_size_test)
        print('-' * 21)
        print('Distillation_acc_test')
        print(global_model.epoch_acc)


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    disalign()



