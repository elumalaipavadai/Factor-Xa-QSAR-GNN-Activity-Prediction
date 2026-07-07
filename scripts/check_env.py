import sys
print("Python", sys.version.split()[0])

import numpy
print("NumPy", numpy.__version__)

from rdkit import Chem
from rdkit import rdBase
print("RDKit", rdBase.rdkitVersion)

from chembl_webresource_client.new_client import new_client
print("ChEMBL client OK")

import torch
print("Torch", torch.__version__)

import transformers
print("Transformers", transformers.__version__)

import deepchem as dc
print("DeepChem", dc.__version__)

from tdc.single_pred import ADME
print("PyTDC/tdc OK")

import torch_geometric
print("PyG", torch_geometric.__version__)

print("\nFXa portfolio environment is ready.")