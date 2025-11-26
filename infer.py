import torch
from models.hyper_mdr import HyperMDR
from utils.energy import unknown_energy

def infer():
    model = HyperMDR()
    model.eval()

    visual = torch.randn(50,512)
    text = torch.randn(10,512)
    proto = torch.randn(20,512)
    state = torch.randn(1,128)
    hx = (torch.zeros(1,128), torch.zeros(1,128))

    logits, h, A, hx = model(visual, text, proto, state, hx)

    score = unknown_energy(logits)
    print("Unknown score:", score.mean().item())

if __name__ == "__main__":
    infer()
