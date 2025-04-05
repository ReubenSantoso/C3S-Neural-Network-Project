import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels1, hidden_channels2, hidden_channels3):
        """
          Behavior: Initialize the Graph Convolutional Neural Network layers with distinct number of neurons.
          Since our class is inheiriting a Pytorch neural network module, we first need to initialize the base class torch.nn.Module to be able to
          gain access to Pytorch methods that will help us use this class to manage and build a neural network.
          Then we initialize our first layer which has an input layer of 17 (because each atom has 17 descriptors), then a middle layer
          with 256 nuerons, then second middle layer goes from 256 -> 128, then last middle layer goes from 128 -> 64 nuerons.
          Then we have a fully connected linear layer to process molecular features (mz, adduct) and we send those to a
          layer with 64 neurons. Lastly, we define a output layer which will a single scalar value (ccs), and since we concatenate the neurons
          from the atomic features (GCN) and graph-level features (Linear), we must multiply by 2.

          Exceptions: None

          Returns: None

          Parameters:
          - hiddden_channels1: the number of neurons in the 1st hidden layer
          - hiddle_channels2: the number of neurons in the 2nd hidden layer
          - hiddle_channels3: the number of neurons in the 3rd hidden layer
        """
        super(GNN, self).__init__()
        self.input = GCNConv(16, hidden_channels1)
        self.hidden1 = GCNConv(hidden_channels1, hidden_channels2)
        self.hidden2 = GCNConv(hidden_channels2, hidden_channels3)

        # Output layer
        self.output = Linear(hidden_channels3, 1)
        self.dropout_rates = [0.5, 0.3, 0.1]

    def forward(self, x, edge_index, batch):
        """
          Behavior: this method defines how data flows through the network.

          Exceptions: None

          Returns:
           - int: The predicted ccs value for a molecule

          Parameters:
           - x: a vector of node features for a molecule
           - edge_index: a vector of the bond features for a molecule
           - u: a vector of the molecular features for a molecule
        """

        #applies the atomic and bond vectors to the input layer
        x = self.input(x, edge_index)
        # apply a relu tranformation to introduce non linearity?
        x = F.relu(x)
        # apply a dropout rate of 0.5 to prevent overfitting
        x = F.dropout(x, p=self.dropout_rates[0], training=self.training)

        x = self.hidden1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rates[1], training=self.training)

        x = self.hidden2(x, edge_index)
        x = F.relu(x)
        # apply a dropout rate of 0.1 to prevent overfitting
        x = F.dropout(x, p=self.dropout_rates[2], training=self.training)

        # aggregates node level embeddings into a single graph level embedding (x) by computing the mean of all node
        # embeddings for each graph in the batch.
        x = global_mean_pool(x, batch)

        return self.output(x)