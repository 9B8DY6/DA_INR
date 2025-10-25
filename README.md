# DA-INR (MICCAI 2025 Accepted)

### MICCAI 2025, "Dynamic-Aware Spatio-temporal Representation Learning for Dynamic MRI Reconstruction"

[MICCAI 2025 Open Access Paper Link](https://papers.miccai.org/miccai-2025/0269-Paper5140.html)

**Dayoung Baik**, Jaejun Yoo*  <br />
*corresponding author

![Alt text](./readme_img/method.jpg)

## Abstract
Dynamic MRI reconstruction, one of inverse problems, has seen a surge by the use of deep learning techniques. Especially, the practical difficulty of obtaining ground truth data has led to the emergence of unsupervised learning approaches. A recent promising method among them is implicit neural representation (INR), which defines the data as a continuous function that maps coordinate values to the corresponding signal values. This allows for filling in missing information only with incomplete measurements and solving the inverse problem effectively. Nevertheless, previous works incorporating this method have faced drawbacks such as long optimization time and the need for extensive hyperparameter tuning. To address these issues, we propose Dynamic-Aware INR (DA-INR), an INR-based model for dynamic MRI reconstruction that captures the spatial and temporal continuity of dynamic MRI data in the image domain and explicitly incorporates the temporal redundancy of the data into the model structure. As a result, DA-INR outperforms other models in reconstruction quality even at extreme undersampling ratios while significantly reducing optimization time and requiring minimal hyperparameter tuning.

## Setting
### Environment
python 3.9, cuda 11.8
1. pytorch install
2. python-nufft install <https://github.com/dfm/python-nufft>
```
git clone https://github.com/dfm/python-nufft
cd python-nufft 
python setup.py install 
```
3. pip install -r requirements.txt
4. torchNGP install <https://github.com/ashawkey/torch-ngp>
    - Rename torch-ngp to torchngp

### Data
Cardiac Cine Data [LINK](https://drive.google.com/file/d/1InP02lh_T_pMPCiwYsavihrcIF4bBVEf/view)

## Command Line
Dynamic MRI reconstruction
```
python3 main.py --down_ratio 1 --sr_ratio 1
```

Spatial Interpolation
```
python3 main.py --spatial_interpolation --sr_ratio 2 --down_ratio 1
```

Temporal Interpolation
```
python3 main.py --temporal_interpolation --sr_ratio 1--down_ratio 2
```

