## Dependencies
- numpy==1.26.1
- torch==2.2.1  
- torch-geometric==2.5.2  
- torch-cluster==1.6.3  
- torch-sparse==0.6.18   
- torch-scatter==2.1.2  


## Usage
##### 1. Install dependencies
```
conda create --name EdgePrompt -y python=3.9.18
conda activate EdgePrompt
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.1 torch-geometric==2.5.2
pip install torch-cluster torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```
##### 2. Run code
For node-level pre-training
```
cd node
python pretrain.py
```
For graph-level pre-training
```
cd graph
python pretrain.py
```

## 
