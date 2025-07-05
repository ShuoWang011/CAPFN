# CAPFN
This repository contains the code and data associated with our paper titled "Few-shot segmentation network based on class-aware prototype fusion." 
### Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14
- ### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

   Download the [data](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/EZboVV33hpZCo670labrD0kBJfqK4bEJHjYFF1ikubFU5A?e=ytsyMx) lists (.txt files) and put them into the `lists` directory. 

- Run `util/get_mulway_base_data.py` to generate base annotations for **stage1**, or directly use the trained weights.
### Usage

- Change configuration via the `.yaml` files in `config`, then run the `.sh` scripts for training and testing.

- **Stage1** *Pre-training*

  Train the base learner within the standard learning paradigm.

  ```
  sh train_base.sh
  ```

- **Stage2** *Meta-training*

  Train the meta learner and ensemble module within the meta-learning paradigm. 

  ```
  sh train.sh
  ```

- **Stage3** *Meta-testing*

  Test the proposed model under the standard few-shot setting. 

  ```
  sh test.sh
  ```

- **Stage4** *Generalized testing*

  Test the proposed model under the generalized few-shot setting. 
  sh test_GFSS.sh
  ```
  ```
  
  
