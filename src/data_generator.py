# src/data_generator.py
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import os

def create_synthetic_codebase_graph(num_files=30, edge_prob=0.1):
    """
    Cria um grafo que simula as dependências em uma codebase.
    """
    # Cria um grafo direcionado para simular importações
    g = nx.DiGraph()
    g.add_nodes_from(range(num_files))

    # Adiciona arestas aleatoriamente
    for i in range(num_files):
        for j in range(num_files):
            if i != j and np.random.rand() < edge_prob:
                # Aresta de i para j significa "i importa j"
                g.add_edge(i, j)

    # Garante que não há nós isolados
    for node in range(num_files):
        if g.in_degree(node) == 0 and g.out_degree(node) == 0:
            target = np.random.randint(0, num_files)
            if node != target:
                g.add_edge(node, target)
    
    # Gera features para os nós (métricas de código simuladas)
    # Feature 1: Linhas de código (normalizadas)
    # Feature 2: Número de funções (normalizadas)
    node_features = np.random.rand(num_files, 2) * 100
    node_features = (node_features - node_features.mean(axis=0)) / (node_features.std(axis=0) + 1e-6)
    x = torch.tensor(node_features, dtype=torch.float)

    # Gera as labels (classes de impacto) baseadas no in-degree (quantos arquivos importam este)
    # Esta é a nossa "verdade fundamental" (ground truth)
    in_degrees = np.array([g.in_degree(n) for n in g.nodes()])
    y = np.zeros(num_files, dtype=int)
    y[in_degrees > 3] = 2  # Alto Impacto (Core)
    y[(in_degrees > 1) & (in_degrees <= 3)] = 1  # Médio Impacto
    y = torch.tensor(y, dtype=torch.long)
    
    # Converte a lista de arestas do NetworkX para o formato do PyG
    edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()

    # Cria o objeto de dados do PyTorch Geometric
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Adiciona máscaras de treino/validação/teste
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:int(0.6*num_nodes)]] = True
    val_mask[indices[int(0.6*num_nodes):int(0.8*num_nodes)]] = True
    test_mask[indices[int(0.8*num_nodes):]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Adiciona nomes de arquivos para a visualização
    data.file_names = [f"module_{i}.py" for i in range(num_files)]
    
    return data, g

if __name__ == '__main__':
    print("Gerando grafo de codebase sintético...")
    os.makedirs('../data', exist_ok=True)
    data, _ = create_synthetic_codebase_graph()
    torch.save(data, '../data/synthetic_codebase.pt')
    print("Grafo salvo em 'data/synthetic_codebase.pt'")
    print(data)
