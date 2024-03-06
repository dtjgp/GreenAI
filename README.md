# Network
 code for final thesis

1. code contains two part for now, first is about the test for the codecarbon lib, second is about test for powermetrics
2. 针对MacBook上的测试暂时告一段落，M1芯片跑alexnet速度实在太慢了，在新的代码中更新了文件路径，其中M1chip文件夹中包括了所有的在MacBook上运行的代码，而Alexnet中的代码为在RTX3060显卡的ubuntu系统上运行的代码以及相关的数据分析及展示
3. 针对论文A First Look into the Carbon Footprint of Federated Learning中提到的psutil进行测试，验证该测试结果中的cpu功耗数据与Codecarbon中的是否一致 2023/11/24

2023/11/25
汇报之后对目前的做法进行调整：
1. 更换更大模型，以便能够进行保存模型参数并且在多个地方进行测试

2024.01.23
1.对 macbook 中的内容和 ubuntu 中的内容进行分开保存
2.在 ubuntu 中，Alexnet_linux 的文件中为最开始的实验内容，用于比较 codecarbon 中的结果和 nvidia-smi 中的结果是否能够对应上（结果：codecarbon 中，对于 GPU 功率的计算没有考虑到改电脑的 GPU 在 Ubuntu 下的功率限制，所以后续的计算以 nvidia-smi 为主）
3.在 GPU 中建立新的文件夹，用于搭建一个 universal 的计算代码

2024.01.24
1. 对 train.py 进行了重新编写，对代码中模糊的命名进行了重新定义

2024.01.25
1. 代码中，前向传播的过程中由于额外的代码开支导致了额外的训练时间，由于目前的计算方法与时间有关，所以需要进行新的调整

2024.01.31
1. 通过对GPU进行同步的代码 'torch.cuda.synchronize()  # 等待数据传输完成'，解决了对每一个部分的时间的精确测量
2. universal.ipynb为模型训练中整体的前向，后向等模块，以及test部分的时间进行测量
3. universal_layers.ipynb为模型训练的前向过程中，对于不同的层（包括卷积，池化等）的时间消耗进行测量
4. train.py and train_layers.py 是前面两个ipynb文件的调用代码

2024.03.03
1. 之前在自己的联想拯救者y9000p（GPU：RTX3060）上运行的数据结果均保存在了universal/Previous_data中
2. 所有的新运行的数据（在远端的RTX4090）上跑的数据均保存在了Data文件夹中