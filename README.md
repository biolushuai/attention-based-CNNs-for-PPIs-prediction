# Protein-Protein Interaction Site Prediction Based on Attention Mechanism and Convolutional Neural Networks
This repository is the implementation of the BIBM2021 paper: "Attention-based convolutional neural networks for protein-protein interaction site prediction"
# Abstract
Abstract—Proteins usually perform their cellular functions by interacting with other proteins. Accurate identification of protein-protein interaction sites (PPIs) from sequence is import for designing new drugs and developing novel therapeutics. A lot of computational models for PPIs prediction have been developed because experimental methods are slow and expensive. Most models employ a sliding window approach in which local neighbors are concatenated to present a target residue. However, those neighbors are not been distinguished by pairwise information between a neighbor and the target. In this study, we propose a novel PPIs prediction model AttCNNPPISP, which combines attention mechanism and convolutional neural networks (CNNs). The attention mechanism dynamically captures the pairwise correlation of each neighbor-target pair within a sliding window, and therefore makes a better understanding of the local environment of target residue. And then, CNNs take the local representation as input to make prediction. Experiments are employed on several public benchmark datasets. Compared with the state-of-the-art models, AttCNNPPISP significantly improves the prediction performance. Also, the experimental results demonstrate that the attention mechanism is effective in terms of constructing comprehensive context information of target residue.

# 1. Datasets
Baidu Netdisk: https://pan.baidu.com/s/1Vm100xiSMJ5PP_SWkUTtwA (Password: PPIS)

# 2. Requirement
Python >=3.8  
Pytorch >=1.10

# 3. Reproducibility

# 4. References
[1] Min Zeng, Fuhao Zhang, Fang-Xiang Wu, Yaohang Li, Jianxin Wang, Min Li*. Protein-protein interaction site prediction through combining local and global features with deep neural networks[J]. Bioinformatics. DOI:10.1093/bioinformaticsz699.  

[2] Bas Stringer*, Hans de Ferrante, Sanne Abeln, Jaap Heringa, K. Anton Feenstra and Reza Haydarlou* (2022). PIPENN: Protein Interface Prediction from sequence with an Ensemble of Neural Nets[J]. Bioinformatics. DOI:10.1093/bioinformatics/btac071.

# 5. Citation
[1] Shuai Lu, Yuguang Li, Xiaofei Nan*, Shoutao Zhang*. Attention-based Convolutional Neural Networks for Protein-Protein Interaction Site Prediction[C]. The 2021 IEEE International Conference on Bioinformatics and Biomedicine(BIBM2021), 2021, 141-144. DOI:10.1109/BIBM52615.2021.9669435.

# 6. Contact
For questions and comments, feel free to contact : ielushuai@126.com.
