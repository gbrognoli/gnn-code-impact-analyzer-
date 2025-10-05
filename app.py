# app.py
import streamlit as st
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

from src.gnn_model import GCN

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Impacto de Código com GNN", layout="wide")
st.title("🚀 Analisador de Impacto de Código com GNNs")
st.write("""
Esta aplicação demonstra como uma Graph Neural Network (GNN) pode prever o impacto de mudanças
em uma base de código. O grafo representa arquivos e suas importações. O modelo prevê se um
arquivo tem impacto **Baixo**, **Médio** ou **Alto** com base em suas características e conexões.
""")

# --- Carregamento de Dados e Modelo (cache para performance) ---
@st.cache_resource
def load_model_and_data():
    try:
        device = torch.device('cpu')
        data = torch.load('data/synthetic_codebase.pt', map_location=device)
        
        num_features = data.num_node_features
        num_classes = len(data.y.unique())
        
        model = GCN(num_node_features=num_features, num_classes=num_classes)
        model.load_state_dict(torch.load('models/gcn_model.pth', map_location=device))
        model.eval()
        
        return model, data
    except FileNotFoundError:
        st.error("Erro: Arquivos de modelo ('models/gcn_model.pth') ou dados ('data/synthetic_codebase.pt') não encontrados.")
        st.info("Por favor, execute 'python src/data_generator.py' e 'python src/train.py' primeiro.")
        return None, None

model, data = load_model_and_data()

if model and data:
    # --- Executa a predição ---
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)

    impact_map = {0: 'Baixo Impacto', 1: 'Médio Impacto', 2: 'Alto Impacto (Core)'}
    color_map = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c'} # Verde, Amarelo, Vermelho

    # --- Interface do Usuário ---
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
        
        # Converte para NetworkX para visualização
        vis_graph = to_networkx(data, to_undirected=False)
        
        node_colors = [color_map[pred.item()] for pred in predictions]
        
        # Desenha o grafo
        fig, ax = plt.subplots(figsize=(12, 10))
        
        pos = nx.spring_layout(vis_graph, seed=42)
        nx.draw_networkx_nodes(vis_graph, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(vis_graph, pos, arrowstyle='->', arrowsize=15, alpha=0.6, ax=ax)
        
        # Highlight no nó selecionado
        nx.draw_networkx_nodes(vis_graph, pos, nodelist=[selected_node_idx], node_color='cyan', edgecolors='black', node_size=700)
        
        # Labels
        labels = {i: name for i, name in enumerate(data.file_names)}
        nx.draw_networkx_labels(vis_graph, pos, labels=labels, font_size=8, font_color='black')
        
        ax.set_title("Grafo da Codebase - Cores por Impacto Previsto")
        plt.axis('off')
        st.pyplot(fig)
