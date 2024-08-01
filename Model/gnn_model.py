from pyspark.sql import SparkSession
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj
from torch.nn import Linear

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GNNRecommendationSystem") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Reviews") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Recommendations") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "200s") \
    .getOrCreate()

print("Spark session initialized.")  # Log to indicate Spark session is ready

# Load and preprocess data
print("Loading and preprocessing data...")

# Load data from MongoDB
df_reviews = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

# Show initial data to check the structure
df_reviews.show()

# Index the user_id and gmap_id columns for use in graph construction
from pyspark.ml.feature import StringIndexer

print("Indexing user and restaurant columns...")

# Convert user_id to numeric indices
user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index").fit(df_reviews)
# Convert gmap_id to numeric indices
rest_indexer = StringIndexer(inputCol="gmap_id", outputCol="rest_id").fit(df_reviews)

df_reviews = user_indexer.transform(df_reviews)
df_reviews = rest_indexer.transform(df_reviews)

# Select only necessary columns and filter out rows with null ratings
df_final = df_reviews.select("user_index", "rest_id", "rating").filter(df_reviews.rating.isNotNull())
df_final = df_final.repartition(200)  # Improve parallel processing performance

# Split data into training and test sets
train, test = df_final.randomSplit([0.8, 0.2])

print("Training data prepared.")  # Log to indicate training data is ready


# Create a NetworkX graph from the Spark DataFrame
def create_networkx_graph(data):
    # Extract edges (user_index, rest_id, rating) from the DataFrame
    edges = data.select("user_index", "rest_id", "rating").rdd.map(tuple).collect()

    # Initialize a bipartite graph
    B = nx.Graph()

    # Add user nodes to the bipartite graph
    users = data.select("user_index").distinct().rdd.map(lambda row: row[0]).collect()
    B.add_nodes_from(users, bipartite=0)
    print("Added user nodes to network.")

    # Add restaurant nodes to the bipartite graph
    restaurants = data.select("rest_id").distinct().rdd.map(lambda row: row[0]).collect()
    B.add_nodes_from(restaurants, bipartite=1)
    print("Added restaurant nodes to network.")

    # Add edges with weights (ratings) to the graph
    B.add_weighted_edges_from(edges)
    print("Added edges with weights to network.")
    return B


# Convert NetworkX graph to PyTorch Geometric Data object
def create_pyg_data(nx_graph):
    # Convert NetworkX graph to edge index tensor
    edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()

    # Extract edge attributes (weights)
    edge_attr = torch.tensor([d['weight'] for (u, v, d) in nx_graph.edges(data=True)], dtype=torch.float).view(-1)

    # Create node feature matrix (identity matrix for simplicity)
    num_nodes = len(nx_graph.nodes)
    x = torch.eye(num_nodes, dtype=torch.float)

    # Create PyTorch Geometric Data object
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data


# Generate NetworkX graph and PyTorch Geometric data
nx_graph = create_networkx_graph(df_final)
pyg_data = create_pyg_data(nx_graph)

print("Graphs created.")  # Log to indicate graph creation is complete


# Define IGMC model using PyTorch Geometric
class IGMC(torch.nn.Module):
    def __init__(self):
        super(IGMC, self).__init__()
        self.rel_graph_convs = torch.nn.ModuleList()
        self.rel_graph_convs.append(RGCNConv(in_channels=4, out_channels=32, num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32, num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32, num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32, num_relations=5, num_bases=4))
        self.linear_layer1 = Linear(256, 128)
        self.linear_layer2 = Linear(128, 1)

    def reset_parameters(self):
        self.linear_layer1.reset_parameters()
        self.linear_layer2.reset_parameters()
        for i in self.rel_graph_convs:
            i.reset_parameters()

    def forward(self, data):
        num_nodes = len(data.x)
        edge_index_dr, edge_type_dr = dropout_adj(data.edge_index, data.edge_type, p=0.2, num_nodes=num_nodes,
                                                  training=self.training)

        out = data.x
        h = []
        for conv in self.rel_graph_convs:
            out = conv(out, edge_index_dr, edge_type_dr)
            out = torch.tanh(out)
            h.append(out)
        h = torch.cat(h, 1)
        h = [h[data.x[:, 0] == True], h[data.x[:, 1] == True]]
        g = torch.cat(h, 1)
        out = self.linear_layer1(g)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear_layer2(out)
        out = out[:, 0]
        return out


model = IGMC()  # Initialize the model


# Early stopping class to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')


# Train the IGMC model
def train_model(pyg_data):
    model = IGMC()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    # Create DataLoader for batching
    train_loader = DataLoader([pyg_data], batch_size=10, shuffle=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, delta=0.001)

    # Training loop
    for epoch in range(100):  # Adjust the number of epochs as needed
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            target = data.edge_attr  # Use actual ratings as targets
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {val_loss}')

        # Check early stopping criteria
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")  # Log early stopping
            break

    # Load the best model weights
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model



# Initialize and train the model
model = train_model(pyg_data)


# Generate recommendations for a given user
def recommend(model, pyg_data, user_index, top_k=10):
    model.eval()

    # Find the user node index
    user_node = user_index

    with torch.no_grad():
        # Get user embedding
        user_embedding = model.rel_graph_convs[0](pyg_data.x[user_node].unsqueeze(0), pyg_data.edge_index)

        # Compute scores for all restaurant nodes
        restaurant_nodes = [n for n in range(pyg_data.num_nodes) if n != user_node]
        restaurant_embeddings = model.rel_graph_convs[0](pyg_data.x[restaurant_nodes], pyg_data.edge_index)

        # Calculate dot product between user embedding and restaurant embeddings
        scores = (restaurant_embeddings @ user_embedding.T).squeeze()
        # Get indices of top-k restaurants
        top_k_indices = scores.argsort(descending=True)[:top_k]

        # Get top-k restaurant node indices
        top_k_restaurants = [restaurant_nodes[i] for i in top_k_indices]

    return top_k_restaurants


# Example usage: recommend top 10 restaurants for a user with index 0
user_index = 0
top_k_restaurants = recommend(model, pyg_data, user_index, top_k=10)
print(f'Top {len(top_k_restaurants)} restaurant recommendations for user {user_index}: {top_k_restaurants}')


# Save recommendations to MongoDB
# def save_recommendations(user_index, top_k_restaurants):
#     recommendations = [{'user_index': user_index, 'rest_id': rest_id} for rest_id in top_k_restaurants]
#     recommendations_df = spark.createDataFrame(recommendations)
#     recommendations_df.write.format("mongo").mode("append").save()
#
#
# # Save recommendations to MongoDB
# save_recommendations(user_index, top_k_restaurants)
# print(f"Recommendations for user {user_index} saved to MongoDB.")

# Stop the Spark session
spark.stop()
print("Spark session stopped.")  # Log to indicate the end of the session
