from utils import random_walk,construct_graph
import math
from tqdm import tqdm
import networkx as nx
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
import torch
from torch.utils.tensorboard import SummaryWriter

import time
import logging

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


##  TODO: 
class TripletDataset(Dataset):

    def __init__(self, graph, walks, nodes2id):
        self.graph = graph
        self.walks = walks
        self.nodes2id = nodes2id

        self.positive_neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes()}
        self.all_nodes = list(graph.nodes())

    def __len__(self):
        return len(self.node_list)
    
    def __getitem__(self, index):
        # Randomly select a anchor node
        walk = self.walks[index]
        anchor = random.choice(walk)

        # Positive sample: a neighbor of the anchor
        positive = random.choice(self.positive_neighbors[anchor])

        # Neigative sample: a node not connected to the anchor node
        while True:
            negative = random.choice(self.all_nodes)
            if not self.graph.has_edge(anchor, negative):
                break

        # Convert nodes to indices
        anchor_idx = self.nodes2id[anchor]
        positive_idx = self.nodes2id[positive]
        negative_idx = self.nodes2id[negative]

        return anchor_idx, positive_idx, negative_idx
        
class NodeEmbeddingModel(nn.Module):

    def __init__(self, num_nodes, embedding_dim):
        super(NodeEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, nodes):
        x = self.embeddings(nodes)
        x = self.fc(x)
        return x
    
def train_triplet_loss_with_visualization(model, dataloader, optimizer, criterion, id2title, num_epochs=10, log_dir='./runs'):

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Get embeddings
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            # Triplet loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss = {epoch_loss:.4f}")

    # 每 5 个 epoch 可视化一次嵌入到 TensorBoard
    if (epoch + 1) % 5 == 0:
        model.eval()
        embeddings = model.embeddings.weight.detach().cpu().numpy()
        writer.add_embedding(
            torch.tensor(embeddings), metadata=id2title, tag=f"Epoch {epoch+1}"
        )
        model.train()

    writer.close()
    logging.info("TensorBoard embeddings logged.")


if __name__=="__main__":
    PATH = "data/ml-latest-small/"
    logging.info("Constructing graph")
    movies_graph, movies = construct_graph(PATH + "movies.csv", PATH + "ratings.csv")    # 构建 图
    logging.info("Sampling walks")
    walks = random_walk(movies_graph,5,10,1,1)    # 随机游走
    nodes2id = dict(zip(movies_graph.nodes(),range(len(movies_graph.nodes()))))
    id2nodes = list(movies_graph.nodes())
    id2title = [movies[movies.movieId==idx].iloc[0].title for idx in id2nodes]
    ##  TODO: 

    dataset = TripletDataset(movies_graph, walks, nodes2id)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_nodes = len(nodes2id)
    embedding_dim = 64
    model = NodeEmbeddingModel(num_nodes, embedding_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(margin = 1.0)

    logging.info("Training loss with TensorBoard visualization")
    train_triplet_loss_with_visualization(
        model, dataloader, optimizer, criterion, id2title, num_epochs=10, log_dir="./runs"
    )