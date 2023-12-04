class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # layer 1
        x = F.relu(self.conv1(x, edge_index))

        # layer 2
        x = F.relu(self.conv2(x, edge_index))

        # layer 3
        x = F.relu(self.conv3(x, edge_index))

        # Global mean pooling to obtain a graph-level representation
        x = torch_geometric.nn.global_mean_pool(x, batch)

        # Fully connected layer for prediction
        x = self.fc(x)

        return x
