# Analisador de Impacto de Código com Graph Neural Networks (GNNs)

Este projeto é uma aplicação web construída com Streamlit e PyTorch Geometric para demonstrar o uso de GNNs em um problema prático de engenharia de software: prever o impacto de uma mudança em uma base de código.

## 🚀 O Problema

Em projetos de software, entender o "raio de explosão" de uma alteração em um arquivo é crucial para testes eficientes e para evitar bugs. Um arquivo pode ser pequeno, mas se for uma dependência central para muitos outros, seu impacto é altíssimo.

Este projeto modela a codebase como um grafo, onde:
* **Nós**: São os arquivos de código.
* **Arestas**: Representam as importações entre os arquivos.

A GNN aprende a classificar cada arquivo em uma categoria de impacto (Baixo, Médio, Alto) com base não apenas em suas métricas, mas principalmente em sua posição e conectividade dentro do grafo.

## 🛠️ Tecnologias Utilizadas

* **PyTorch Geometric (PyG):** Para a criação e treinamento do modelo GNN.
* **Streamlit:** Para criar a interface web interativa.
* **NetworkX & Matplotlib:** Para a visualização do grafo de dependências.
* **PyTorch:** Como framework base de deep learning.

## ⚙️ Como Executar o Projeto

Siga os passos abaixo para rodar a aplicação localmente.

### 1. Clone o Repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd gnn-code-impact-analyzer
```

### 2. Crie um Ambiente Virtual (Recomendado)

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Gere os Dados e Treine o Modelo

Primeiro, vamos gerar nosso grafo de codebase sintético e depois treinar o modelo GNN.

```bash
# Gera o arquivo data/synthetic_codebase.pt
python src/data_generator.py

# Treina o modelo e salva em models/gcn_model.pth
python src/train.py
```

### 5. Execute a Aplicação Streamlit

Agora, inicie o servidor do Streamlit.

```bash
streamlit run app.py
```

Abra seu navegador no endereço local fornecido (geralmente `http://localhost:8501`) e explore a aplicação!
