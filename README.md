# icdar_vqa

Implementation in Pytorch for ICDAR challege, 2019. 
Paper -- https://arxiv.org/abs/1905.13648
Results -- https://rrc.cvc.uab.es/?ch=11&com=evaluation&task=1

The goal of this repository is to provide an implementation for all the 3 tasks of this challenge namely strongly contexualized, weakly contextualized and open dictionary.

## Prerequisites
- python 3.6+
- numpy
- [pytorch](http://pytorch.org/) 0.4
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [nltk](http://www.nltk.org/install.html)
- [pandas](https://pandas.pydata.org/)

## Data
- Relesed by challenge organizors.
- [COCO 36 features pretrained resnet model](https://github.com/peteanderson80/bottom-up-attention#pretrained-features)
- [GloVe pretrained Wikipedia+Gigaword word embedding](https://nlp.stanford.edu/projects/glove/)

## Steps to use this repository
- First, download the data from challenge website https://rrc.cvc.uab.es/?ch=11&com=downloads

- Secondly, the following command is used to prepare data for training. 
```bash
python prepro.py
  ```
-Once the data is ready for training, the training can be started using this command. 
```bash
bash train.sh
  ```
  
## Notes
-The results are reported in https://arxiv.org/abs/1905.13648. 
-We got 3rd position in 1st task. 
- Some of `preproc.py` and `utils.py` are based on [this repo](https://github.com/markdtw/vqa-winner-cvprw-2017) 

## Resources
- [The paper](https://arxiv.org/pdf/1708.02711.pdf).
- [Their CVPR Workshop slides](http://cs.adelaide.edu.au/~Damien/Research/VQA-Challenge-Slides-TeneyAnderson.pdf).
