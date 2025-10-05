# app.py (versão final corrigida)
import streamlit as st
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import os
import time

# Importa as funções dos outros arquivos
from src.gnn_model import GCN
from src.data_generator import create_synthetic_codebase_graph
from src.train import train as train_model

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Impacto de Código com GNN", layout="wide")
st.title("🚀 Analisador de Impacto de Código com GNNs")
st.write("""
Esta aplicação demonstra como uma Graph Neural Network (GNN) pode prever o impacto de mudanças
em uma base de código. O grafo representa arquivos e suas importações. O modelo prevê se um
arquivo tem impacto **Baixo**, **Médio** ou **Alto** com base em suas características e conexões.
""")

# --- Função para garantir que dados e modelo existam ---
@st.cache_resource(show_spinner="Configurando o ambiente pela primeira vez...")
def setup_and_load():
    """
    Verifica se os arquivos de dados e modelo existem. Se não, os cria.
    Depois, carrega e retorna o modelo e os dados.
    """
    DATA_PATH = 'data/synthetic_codebase.pt'
    MODEL_PATH = 'models/gcn_model.pth'
    
    # Se os arquivos não existem, cria eles
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        st.info("Arquivos não encontrados. Gerando dados e treinando o modelo. Isso pode levar um minuto...")
        
        # Garante que os diretórios existam
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # 1. Gerar dados
        data, _ = create_synthetic_codebase_graph()
        torch.save(data, DATA_PATH)
        
        # 2. Treinar modelo
        train_model()
        # Adiciona uma pequena pausa para garantir que o arquivo seja salvo no disco
        time.sleep(2)

    # Carrega os arquivos
    device = torch.device('cpu')
    
    # ===================== A CORREÇÃO ESTÁ AQUI =====================
    # Adicionamos 'weights_only=False' porque nosso arquivo de dados
    # contém um objeto Python complexo (Data), e não apenas pesos.
    data = torch.load(DATA_PATH, map_location=device, weights_only=False)
    # ================================================================
    
    num_features = data.num_node_features
    num_classes = len(data.y.unique())
    
    model = GCN(num_node_features=num_features, num_classes=num_classes)
    # Para carregar o state_dict, o padrão weights_only=True é seguro e correto.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    return model, data

# --- Lógica principal do App ---
try:
    model, data = setup_and_load()

    # Executa a predição
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)

    impact_map = {0: 'Baixo Impacto', 1: 'Médio Impacto', 2: 'Alto Impacto (Core)'}
    color_map = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c'}

    # Interface do Usuário
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Análise do Grafo")
        selected_file = st.selectbox(
            "Selecione um arquivo para inspecionar:",
            options=data.file_names
        )
        selected_node_idx = data.file_names.index(selected_file)
        predicted_class = predictions[selected_node_idx].item()
        st.markdown(f"Arquivo: **{selected_file}**")
        st.markdown(f"Impacto Previsto: <span style='color:{color_map[predicted_class]}; font-weight:bold;'>{impact_map[predicted_class]}</span>", unsafe_allow_html=True)
        st.subheader("Legenda de Impacto")
        for i in range(len(impact_map)):
            st.markdown(f"<span style='color:{color_map[i]};'>●</span> {impact_map[i]}", unsafe_allow_html=True)
        st.subheader("Métricas do Arquivo (Simuladas)")
        st.text(f"Linhas de Código (norm): {data.x[selected_node_idx][0]:.2f}")
        st.text(f"Nº de Funções (norm): {data.x[selected_node_idx][1]:.2f}")
    with col2:
        st.header("Visualização do Grafo de Dependências")
        vis_graph = to_networkx(data, to_undirected=False)
        node_colors = [color_map[pred.item()] for pred in predictions]
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(vis_graph, seed=42)
        nx.draw_networkx_nodes(vis_graph, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(vis_graph, pos, arrowstyle='->', arrowsize=15, alpha=0.6, ax=ax)
        nx.draw_networkx_nodes(vis_graph, pos, nodelist=[selected_node_idx], node_color='cyan', edgecolors='black', node_size=700)
        labels = {i: name for i, name in enumerate(data.file_names)}
        nx.draw_networkx_labels(vis_graph, pos, labels=labels, font_size=8, font_color='black')
        ax.set_title("Grafo da Codebase - Cores por Impacto Previsto")
        plt.axis('off')
        st.pyplot(fig)

except Exception as e:
    st.error(f"Ocorreu um erro ao inicializar o aplicativo: {e}")
    st.info("Verifique se todas as dependências estão instaladas corretamente e tente recarregar a página.")
