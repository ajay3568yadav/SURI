import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
from torch_geometric.loader import DataLoader as Loader
from dataloader import DataLoader  # assuming this import is necessary
from delayobject import DelayObject



# Loading the object_dictionary containing information of DelayObjects
with open('../newPickle/object_dictionary.pickle', 'rb') as f:
    object_dictionary = pickle.load(f)

# Loading the data_loader_list containing information of DataLoader Objects
with open('data_loader_list.pkl', 'rb') as file:
    data_loader_list = pickle.load(file)
    

# Maximum length of the padded node_feature list
max_len = 29692
        
# Set of all Uniquie cells
cell_set = set()
for data in data_loader_list:
    for nodes in data.graph:
        cell_set.add(nodes.cell)

# Dictionary mapping cells to their respective index for Label Encoding
cell_labels = {}
       
# Label Encoding
for index,cell in enumerate(cell_set):
    cell_labels[cell] = (index - 1)/(len(cell_set)-1)

# Reducing Dimensions with Principal Component Analysis(PCA)
pca = PCA(n_components=100)
scaler = StandardScaler()

# Returns the Switching time list using the node's radix mapped to DelayObject in object_dictionary
def get_switching_time(node):
    return list(object_dictionary[node.radix][0].toggle.keys())

# Returns the Node Features Matrix of the Graph
def get_node_feats(graph):
    
    node_feats = []
    # Iterating through the list of nodes in the Graph 
    for node in graph:
        
        # Getting the corresponding numerical label from the cell_labels dictionary
        cell_label = cell_labels.get(node.cell)

        switching_time = list(node.powertimedict.keys())
        
        switching = switching_time + (max_len - len(switching_time)) * [0]
        
        node_feats.append([cell_label, node.peak] + switching)
    
    # Convert node_feats to a NumPy array
    node_feats = np.array(node_feats)
    
    # Extract the last columns
    node_feats_last_columns = node_feats[:, -max_len:]

    # Set the number of components to reduce to
    node_feats_last_columns_pca = pca.fit_transform(node_feats_last_columns)
    
    # Normalize the combined node features
    node_feats_combined = np.hstack((node_feats[:, :-max_len], node_feats_last_columns_pca))
    node_feats_combined = scaler.fit_transform(node_feats_combined)

    return node_feats_combined

# Returns the Labels for the Corresponding Graph
def get_label(label_dict):
    keys = list(label_dict.keys())
    values = list(label_dict.values())
    
    # Convert keys and values to tensors
    keys_tensor = torch.tensor(keys, dtype=torch.float32).unsqueeze(1)
    values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    
    # Concatenate keys and values vertically
    label_tensor = torch.cat([keys_tensor, values_tensor], dim=1)
    
    reshaped_label_tensor = label_tensor.view(500, 2)
    
    return reshaped_label_tensor


# Create a Dataset for Training, containing pytorch_geometric.data.Data Objects
dataset = []
for data in data_loader_list:
    if len(data.graph) >= 100 and len(data.graph) <= 1000:
        # Convert x and y to tensors with dtype float32
        x = torch.tensor(get_node_feats(data.graph), dtype=torch.float32)
        y = get_label(data.label)
        edge_index = torch.LongTensor(data.edge_index)
        
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
        print(len(dataset))
        



# Iterate over the dataset
for data in dataset:
    # Convert keys and values to arrays
    key_array = data.y[:,0]
    value_array = data.y[:,1]

    # Scale key array
    scaler = MinMaxScaler()
    scaled_key_array = scaler.fit_transform(key_array.reshape(-1, 1))

    # Scale value array
    scaled_value_array = scaler.fit_transform(value_array.reshape(-1, 1))

    # Concatenate scaled arrays along the second axis
    data.y = np.concatenate((scaled_key_array, scaled_value_array), axis=1)

    
#Testing with lower number of label time points
for data in dataset:
    data.yy = torch.as_tensor(data.y[::5], dtype=torch.float32)


subset = []
for data in dataset:
    for time_power_list in data.yy:
        # Create a new Data object and copy relevant attributes
        data_copy = Data(x=data.x.clone(), edge_index=data.edge_index.clone(), yy=time_power_list[1])

        # Add or modify the 'time' attribute
        data_copy.time = torch.tensor([time_power_list[0]] * len(data.x),dtype = torch.float32)

        subset.append(data_copy)
    if len(subset)>=100000:
        break
    if len(subset)%10000 == 0:
        print(len(subset))


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(input_dim + 1, hidden_dim)  # +1 for time feature
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        time = data.time.view(-1, 1)  # Reshape time to [num_nodes, 1]

        # Concatenate time feature to node features
        x = torch.cat((x, time), dim=1)

        # Apply GraphConv layers normally
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layer
        x = F.relu(self.fc(x))  # Applying activation function

        # Reshape x to [num_graphs, output_dim]
        x = x.view(-1, output_dim)

        return x



train_ratio = 0.8
num_train = int(len(subset) * train_ratio)
num_test = len(subset) - num_train

# Using random_split to create training and testing datasets
train_dataset, test_dataset = random_split(subset, [num_train, num_test])

# Create DataLoader instances for training and testing
train_loader = Loader(train_dataset, batch_size=96, shuffle=True, drop_last=True, num_workers=2)
test_loader = Loader(test_dataset, batch_size=96, shuffle=False, drop_last=True, num_workers=2)


input_dim = 102
hidden_dim = 200
output_dim = 1
model = GNNModel(input_dim, hidden_dim, output_dim)

# Using Mean Squared Error Loss for regression
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1000


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)

        # Flatten the outputs for each data point in the batch
        outputs_flat = outputs.view(-1)

        # Flatten and reshape the labels for each data point in the batch
        labels = batch.yy.float()  # Cast to torch.float

        loss = criterion(outputs_flat, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], 
                  Loss: {loss.item():.4f}')

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {average_loss:.4f}')
    

# Test loop
model.eval()
with torch.no_grad():
    total_test_loss = 0.0
    predictions = []
    true_labels = []

    for test_batch in test_loader:
        test_batch = test_batch.to(device)
        test_outputs = model(test_batch)

        # Flatten the outputs for each data point in the batch
        test_outputs_flat = test_outputs.view(-1)

        # Flatten and reshape the labels for each data point in the batch
        test_labels = test_batch.yy.float()

        test_loss = criterion(test_outputs_flat, test_labels)
        total_test_loss += test_loss.item()

        # Save predictions and true labels for further analysis if needed
        predictions.extend(test_outputs_flat.cpu().numpy())
        true_labels.extend(test_labels.cpu().numpy())
        
        print(f'Test Loss: {test_loss.item():.4f}')

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss:.4f}')
                  
                  
