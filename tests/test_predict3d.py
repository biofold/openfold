import torch
import sys
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Create compatibility layer for torch._six
if not hasattr(torch, '_six'):
    class TorchSix:
        inf = float('inf')
        string_classes = (str, bytes)
        
        @classmethod
        def container_abcs(cls):
            import collections.abc as container_abcs
            return container_abcs
    
    torch._six = TorchSix
    # Also patch sys.modules in case deepspeed looks for it there
    sys.modules['torch._six'] = TorchSix


import esm
print(f"ESM version: {esm.__version__}")

# Test ESMFold
model = esm.pretrained.esmfold_v1()
#model = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().cuda()  # Now we can use GPU!
#model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#batch_converter = alphabet.get_batch_converter()
#print("ESMFold model loaded on GPU")

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

with torch.no_grad():
    output = model.infer_pdb(sequence)

with open("result.pdb", "w") as f:
    f.write(output)

import biotite.structure.io as bsio
struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
print(f"pLDDT: {struct.b_factor.mean():.1f}")
