# FEDIC: Federated Learning on Non-IID and Long-tailed Data via Calibrated Distillation

This is the code for paper: **FEDIC: Federated Learning on Non-IID and Long-tailed Data via Calibrated Distillation.**

**Abstract:** Dealing with non-IID data is one of the most challenging problems for federated learning. Researchers have proposed a variety of methods to try to eliminate the negative influence of non-IIDness. However, they only focus on the non-IID data but generally assume that the universal class distribution is balanced. In many real-world applications, the universal class distribution is long-tailed such that the model will be highly biased to the head classes and performs poorly on the tail classes, although some methods can alleviate the problem of non-IID data. Therefore, this paper studies the joint problem of non-IID and long-tailed data in federated learning and proposes a corresponding solution called Federated Ensemble Distillation with Imbalance Calibration (FEDIC). To deal with non-IID data, FEDIC uses model ensemble to take advantage of the diversity of models trained on non-IID data. Then, a new distillation method with client-wise and class-wise logit adjustment and calibration gating network is proposed to solve the long-tail problem effectively. We evaluate FEDIC on three long-tailed datasets (CIFAR-10-LT, CIFAR-100-LT, and ImageNet-LT) with the non-IID experimental setting, compared with state-of-the-art methods of federated learning and long-tail learning. Specifically designed experiments also validate the effectiveness of the proposed calibrated distillation method. 



### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4



### Dataset

- CIFAR-10
- CIFAR-100
- ImageNet-LT



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                    | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `num_classes`               | Number of classes                                            |
| `num_clients`               | Number of all clients.                                       |
| `num_online_clients`        | Number of participating local clients.                       |
| `num_rounds`                | Number of communication rounds.                              |
| `num_data_train`            | Number of training data.                                     |
| `num_epochs_local_training` | Number of local epochs.                                      |
| `batch_size_local_training` | Batch size of local training.                                |
| `server steps`              | Number of steps of  training calibrated network.             |
| `distillation steps`        | Number of distillation steps.                                |
| `lr_global_teaching`        | Learning rate of server updating.                            |
| `lr_local_training`         | Learning rate of client updating.                            |
| `non_iid_alpha`             | Control the degree of non-IIDness.                           |
| `imb_factor`                | Control the degree of imbalance.                             |
| `ld`                        | Control the trade-off between $L_{CE}$ and $\lambda L_{KL}.$ |



### Usage

Here is an example to run FEDIC on CIFAR-10 with imb_factor=0.01:

```python
python main.py --num_classrs=10 \ 
--num_clients=20 \
--num_online_clients=8 \
--num_rounds=200 \
--num_data_training=49000 \
--num_epochs_local_training=10 \
--batch_size_local_training=128 \
--server_steps=100 \
--distillation_steps=100 \
--lr_global_training=0.001 \
--lr_local_training=0.1 \
--non-iid_alpha=0.1 \
--imb_factor=0.01 \ 
--ld=0.5
```



