# Network
 code for final thesis

1. code contains two part for now, first is about the test for the codecarbon lib, second is about test for powermetrics
2. 针对MacBook上的测试暂时告一段落，M1芯片跑alexnet速度实在太慢了，在新的代码中更新了文件路径，其中M1chip文件夹中包括了所有的在MacBook上运行的代码，而Alexnet中的代码为在RTX3060显卡的ubuntu系统上运行的代码以及相关的数据分析及展示
3. 针对论文A First Look into the Carbon Footprint of Federated Learning中提到的psutil进行测试，验证该测试结果中的cpu功耗数据与Codecarbon中的是否一致 2023/11/24

2023/11/25
汇报之后对目前的做法进行调整：
1. 更换更大模型，以便能够进行保存模型参数并且在多个地方进行测试