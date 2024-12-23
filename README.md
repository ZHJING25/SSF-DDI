## SSF-DDI: A Deep Learning Method Utilizing Drug Sequence and Substructure  


## 安装依赖
numpy==1.18.1 \
tqdm==4.42.1 \
pandas==1.0.1 \
rdkit==2009.Q1-1 \
scikit_learn==1.0.2 \
torch==1.11.0 \
torch_geometric==2.0.4 \
torch_scatter==2.0.9

## 模型训练:  
### 1. DrugBank
```python
    python data_preprocessing.py -d drugbank -o all #数据处理
    python train.py --fold 0 --save_model #模型训练
```

### 2. TWOSIDES
```python
    python data_preprocessing.py -d twosides -o all #数据处理
    python train.py --fold 0 --save_model #模型训练
```

## 模型测试: 
```python
    python get_test_result.py #数据处理
```
update 20241223
