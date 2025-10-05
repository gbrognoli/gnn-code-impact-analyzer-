# src/train.py
import torch
import torch.nn.functional as F
from src.gnn_model import GCN
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carrega os dados
    try:
        data = torch.load('data/synthetic_codebase.pt').to(device)
    except FileNotFoundError:
        print("Arquivo de dados não encontrado. Rode 'python src/data_generator.py' primeiro.")
        return

    # Instancia o modelo
    model = GCN(
        num_node_features=data.num_node_features,
        num_classes=len(torch.unique(data.y))
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')

    # Salva o modelo treinado
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/gcn_model.pth')
    print("\nModelo treinado e salvo em 'models/gcn_model.pth'")

    # Avalia a acurácia no conjunto de teste
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Acurácia no conjunto de teste: {acc:.4f}')

if __name__ == '__main__':
    train()
