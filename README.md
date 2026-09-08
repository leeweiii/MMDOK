# MMDOK
This is a PyTorch implementation of the paper MMDOK: A Multi-modal and Multi-scale Disease-oriented Fusion Framework with Kolmogorov-Arnold Networks

![Overall Framework](./README_img/overallframework.png)

## Install
```
pip install -r requirements.txt
```

## Dataset preparation
```
      Data
      ├── SMT
      │   ├── images
      │   │   ├── Center1
      │   │   ├── Center2
      │   │   ├── Center3
      │   │   ├── ...
      │   ├── list
      │   │   ├── train_label.csv
      │   │   ├── val_label.csv
      │   │   ├── test_label.csv
```

## Train
```
python run.py -c ./configs/custom_config.yaml --train
```

## Infer
```
python run.py -c ./configs/custom_config.yaml --test --ckpt_path=ckpt_path
```

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@ARTICLE{11559629,
  author={Li, Wei and Gong, Xun and Fan, Lin and Li, Xinxin and Li, Jiao and Sun Xiaobin},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={MMDOK: A Multi-modal and Multi-scale Disease-oriented Fusion Framework with Kolmogorov-Arnold Networks}, 
  year={2026},
  volume={},
  number={},
  pages={1-14},
  keywords={Medical image analysis; Representation learning; Multi-modal fusion; Multi-scale alignment; Kolmogorov-Arnold Networks},
  doi={10.1109/JBHI.2026.3703035}}
```
