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

2024.04.04
1. 创建新的cloud文件夹，其中包括了所有的要进行测试的内容：
    1. 首先要测试在同步和非同步下，模型的能耗数据的区别大概有多少区别
    2. 测试不同的模型在不同的硬件条件下，其各自的能耗，用时，以及模型的特性，包括MACs，每次计算中，是否包括类似于resnet的结构（目前考虑resnet结构可能会在同步的情况下，造成大量的时间浪费以及能耗浪费）
2. 在PVWatts文件夹中创建code文件夹，用于保存DatacenterClass.py, StoragebatteryClass.py以及SolarpanelClass.py文件，用于后续对于数据中心的能耗，储能电池以及太阳能板进行建模

2024.04.10
1. 4090_sync_nosync主要的测试内容是sync与nosync的情况下，能耗是否有差距，以及二者之间会有什么样的变化

2024.05.26
1. 3060_95W/Data/googlenet_mod1是去掉了Inception中的p1,mod2是去掉了p2，mod3是去掉了p3，mod4是去掉了p4
2. 3060_95W/Data/googlenet_mod5,mod6,mod7,mod8,mod9是对inception部分进行了调整，将所有的分支都设置为了相同的结构，然后探究并行的影响是怎么样影响的运行时间以及能耗，其中：
    1) mod5是只保留一个分支进行运行
    2) mod6保留两个分支
    3) mod7三个分支 
    4) mod8四个分支
    5) mod9针对不加入任何的inception的时候的能耗进行计算

2024.05.30
1. 增加nosync代码，确认一下并行的效率能够到多少，
    1) nosync1为针对原始googlenet，单独考虑前向传播过程中的优化
    3) nosync3为针对原始googlenet，对多个过程进行合并，看其优化的特性，全部进行nosync处理

2024.06.20
1. Analysis code is used to import all model and get the MACs as well as the Parameters info
2. The FinalDataAnalysis is to show the relationship between the MACs, Parameters and energy consumption(original version)
3. The FinalDataAnalysis_fun1 is the improvement version of FinalDataAnalysis, with all the plot written into function
4. The FinalDataAnalysis_fun2 is the code that try to use the Machine Learning algorithm to analysis the math relation between MACs only with the Energy consumption
5. add a FinalDataAnalysis_fun3 to create a new test dataset

2024.06.24
1. FinalDataAnalysis_fun4主要是针对所有的MACs的参数进行归一化