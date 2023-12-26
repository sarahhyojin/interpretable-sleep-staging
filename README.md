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
- SwinTransformer (ICCV 2021) [[PDF]](https://arxiv.org/abs/2103.14030)
![Intra-Epoch-Learning](https://github.com/sarahhyojin/interpretable-sleep-staging/assets/75148383/78383d74-7ad5-4968-b1b5-f3c1a417e222)


  
### Inter-Epoch Learning
- Transformer Encoder (NIPS 2017) [[PDF]](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
<p align="center"><img width="762" alt="Screenshot 2023-11-07 at 11 13 52 AM" src="https://github.com/sarahhyojin/SwinSleepNet/assets/75148383/f971db10-34c2-4cf4-8efa-d3ca525f926a"></p>

## Interpretability
### Inter-Epoch Interpretability
- Adopted Transformer Interpretability Beyond Attention Visualization (CVPR 2021) [[Github]](https://github.com/hila-chefer/Transformer-Explainability)
- Aggregated 10,000 accurate predictions per class
<p align="center"><img width="760" alt="Screenshot 2023-11-07 at 10 49 10 AM" src="https://github.com/sarahhyojin/SwinSleepNet/assets/75148383/481b2886-c27f-462f-80f6-f3b952da448e"></p>

- Individual predictions
<p align="center"><img width="695" alt="Screenshot 2023-11-07 at 11 43 50 AM" src="https://github.com/sarahhyojin/SwinSleepNet/assets/75148383/0412a452-dab0-4879-b639-f48c12516d47"></p>


### Intra-Epoch Interpretability
<p align="center"><img width="825" alt="Screenshot 2023-11-07 at 11 44 58 AM" src="https://github.com/sarahhyojin/SwinSleepNet/assets/75148383/e8a978e3-4ac9-483f-9acb-cc0e0f547080"></p>
