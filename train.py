import torch
from models.hyper_mdr import HyperMDR
from utils.energy import unknown_energy

def train_loop():
    model = HyperMDR().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    hx = (torch.zeros(1,128).cuda(), torch.zeros(1,128).cuda())

    for step in range(1000):
        visual = torch.randn(50,512).cuda()
        text = torch.randn(10,512).cuda()
        proto = torch.randn(20,512).cuda()
        state = torch.randn(1,128).cuda()

        logits, h, A, hx = model(visual, text, proto, state, hx)
        loss = logits.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} Loss={loss.item():.4f}")

if __name__ == "__main__":
    train_loop()
