# URAMN

The source code for the paper: ”[Unsupervised Representation Learning on Attributed Multiplex Network](https://dl.acm.org/doi/abs/10.1145/3511808.3557486)" accepted in CIKM 2022 by Rui Zhang, Arthur Zimek and Peter Schneider-Kamp.


```
@inproceedings{10.1145/3511808.3557486,
author = {Zhang, Rui and Zimek, Arthur and Schneider-Kamp, Peter},
title = {Unsupervised Representation Learning on Attributed Multiplex Network},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557486},
doi = {10.1145/3511808.3557486},
abstract = {Embedding learning in multiplex networks has drawn increasing attention in recent years and achieved outstanding performance in many downstream tasks. However, most existing network embedding methods either only focus on the structured information of graphs, rely on the human-annotated data, or mainly rely on multi-layer GCNs to encode graphs at the risk of learning ill-posed spectral filters. Moreover, it is also challenging in multiplex network embedding to learn consensus embeddings for nodes across the multiple views by the inter-relationship among graphs. In this study, we propose a novel and flexible unsupervised network embedding method for attributed multiplex networks to generate more precise node embeddings by simplified Bernstein encoders and alternate contrastive learning between local and global. Specifically, we design a graph encoder based on simplified Bernstein polynomials to learn node embeddings of a specific graph view. During the learning of each specific view, local and global contrastive learning are alternately applied to update the view-specific embedding and the consensus embedding simultaneously. Furthermore, the proposed model can be easily extended as a semi-supervised model by adding additional semi-supervised cost or as an attention-based model to attentively integrate embeddings from multiple graphs. Experiments on three publicly available real-world datasets show that the proposed method achieves significant improvements on downstream tasks over state-of-the-art baselines, while being faster or competitive in terms of runtime compared to the previous studies.},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {2610–2619},
numpages = {10},
keywords = {graph representation learning, graph neural network, contrastive learning, attributed multiplex network embedding},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```

## Dataset
Preprocessed datasets can be found on https://www.dropbox.com/s/d1altfstj90ylb2/dataset.rar?dl=0

