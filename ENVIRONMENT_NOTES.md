# FXa GNN Portfolio — Environment Notes



Environment name: fxa\_portfolio\_clean  

Created: June 15, 2026  

Platform: Windows 11, x86-64



## Verified working stack



Pasted from `check\env.py` output:



```Python 3.10.20

NumPy 1.26.4

RDKit 2022.09.5

ChEMBL client OK

Torch 2.12.0+cpu

Transformers 4.50.3

DeepChem 2.8.0

PyTDC/tdc OK

PyG 2.8.0



#To recreate

conda env create -f fxa\_portfolio\_clean\_environment.yml

conda activate fxa\_portfolio\_clean

python check\_env.py

