# Interpretable Automatic Sleep-staging
Clinicians "See" the Waverforms: A Computer Vision Approach for Interpretable Sleep-staging

## Framework Overview
![Framework_Overview](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/ad4a3876-e872-4a42-b6f4-50ef755af149)



## Dataset
Jaemin Jeong, Wonhyuck Yoon, Jeong-Gun Lee, Dongyoung Kim, Yunhee Woo, Dong-Kyu Kim, Hyun-Woo Shin, __Standardized image-based polysomnography database and deep learning algorithm for sleep-stage classification__, *Sleep*, 2023;, zsad242, https://doi.org/10.1093/sleep/zsad242
- [SNUH Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=210): Only Avaliable at Restricted Area
![Dataset](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/8e2e9e00-4cb1-42be-a605-965eb02f8d0b)


## Model Architecture
### Intra-Epoch Learning
- Vision Transformer (ICLR 2021) [[PDF]](https://arxiv.org/pdf/2010.11929.pdf)
![Intra-Epoch-Learning](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/78383d74-7ad5-4968-b1b5-f3c1a417e222)


  
### Inter-Epoch Learning
- Transformer Encoder (NIPS 2017) [[PDF]](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
![Inter-Epoch-Learning](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/f2402300-aa1e-4fa7-ba88-b886b464b8b8)


## Interpretability
### Inter-Epoch Interpretability
- Adopted Transformer Interpretability Beyond Attention Visualization (CVPR 2021) [[Github]](https://github.com/hila-chefer/Transformer-Explainability)
- Baseline (Eigen-CAM) : Jaemin Jeong, Wonhyuck Yoon, Jeong-Gun Lee, Dongyoung Kim, Yunhee Woo, Dong-Kyu Kim, Hyun-Woo Shin, __Standardized image-based polysomnography database and deep learning algorithm for sleep-stage classification__, *Sleep*, 2023;, zsad242, https://doi.org/10.1093/sleep/zsad242
![Intra-Epoch-Explainability](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/180a3767-242b-49c4-b7b7-f6e148a6cf76)


### Intra-Epoch Interpretability
- Adopted same method as above.
- Using sliding window scheme and aggregated 15 softmax values to predict one Epoch.
![Inter-Epoch-Interpretability-4](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/d154bf38-2711-4497-8ed4-3684ba60a79a)
![Inter-Epoch-Interpretability-2](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/a03148db-89e5-481c-82de-bd751515f872)
![Inter-Epoch-Interpretability-1](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/a7d0e031-4b98-449f-abd8-230937d955ef)
