import time
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tf_slim as slim
import numpy as np
import os
import logging

# Initialize TensorFlow session
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )
)

# Load data from MongoDB
print("Loading data from MongoDB...")
# Connect to MongoDB and load data
client = MongoClient('mongodb://localhost:27017/')
db = client['Google-Maps-Restaurant']
collection = db['Reviews']

data = pd.DataFrame(list(collection.find()))
print(data.columns)

# Encode user_id and gmap_id
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

data['user_index'] = user_encoder.fit_transform(data['user_id'])
data['item_index'] = item_encoder.fit_transform(data['gmap_id'])

# Prepare data for TensorFlow
n_users = data['user_index'].nunique()
n_items = data['item_index'].nunique()

print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Set up logging
logger = logging.getLogger(__name__)
MODEL_CHECKPOINT = "model.ckpt"

class NCF:
    """Neural Collaborative Filtering (NCF) implementation"""

    def __init__(self, n_users, n_items, model_type="NeuMF", n_factors=8, layer_sizes=[16, 8, 4],
                 n_epochs=50, batch_size=64, learning_rate=5e-3, verbose=1, seed=None):
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.n_users = n_users
        self.n_items = n_items
        self.model_type = model_type.lower()
        self.n_factors = n_factors
        self.layer_sizes = layer_sizes
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        model_options = ["gmf", "mlp", "neumf"]
        if self.model_type not in model_options:
            raise ValueError("Wrong model type, please select one of this list: {}".format(model_options))

        self.ncf_layer_size = n_factors + layer_sizes[-1]
        self._create_model()
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _create_model(self):
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.variable_scope("input_data", reuse=tf.compat.v1.AUTO_REUSE):
            self.user_input = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])
            self.item_input = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])
            self.labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        with tf.compat.v1.variable_scope("embedding", reuse=tf.compat.v1.AUTO_REUSE):
            self.embedding_gmf_P = tf.Variable(
                tf.random.truncated_normal([self.n_users, self.n_factors], mean=0.0, stddev=0.01, seed=self.seed),
                name="embedding_gmf_P", dtype=tf.float32)

            self.embedding_gmf_Q = tf.Variable(
                tf.random.truncated_normal([self.n_items, self.n_factors], mean=0.0, stddev=0.01, seed=self.seed),
                name="embedding_gmf_Q", dtype=tf.float32)

            self.embedding_mlp_P = tf.Variable(
                tf.random.truncated_normal([self.n_users, int(self.layer_sizes[0] / 2)], mean=0.0, stddev=0.01, seed=self.seed),
                name="embedding_mlp_P", dtype=tf.float32)

            self.embedding_mlp_Q = tf.Variable(
                tf.random.truncated_normal([self.n_items, int(self.layer_sizes[0] / 2)], mean=0.0, stddev=0.01, seed=self.seed),
                name="embedding_mlp_Q", dtype=tf.float32)

        with tf.compat.v1.variable_scope("gmf", reuse=tf.compat.v1.AUTO_REUSE):
            self.gmf_p = tf.reduce_sum(input_tensor=tf.nn.embedding_lookup(params=self.embedding_gmf_P, ids=self.user_input), axis=1)
            self.gmf_q = tf.reduce_sum(input_tensor=tf.nn.embedding_lookup(params=self.embedding_gmf_Q, ids=self.item_input), axis=1)
            self.gmf_vector = self.gmf_p * self.gmf_q

        with tf.compat.v1.variable_scope("mlp", reuse=tf.compat.v1.AUTO_REUSE):
            self.mlp_p = tf.reduce_sum(input_tensor=tf.nn.embedding_lookup(params=self.embedding_mlp_P, ids=self.user_input), axis=1)
            self.mlp_q = tf.reduce_sum(input_tensor=tf.nn.embedding_lookup(params=self.embedding_mlp_Q, ids=self.item_input), axis=1)
            output = tf.concat([self.mlp_p, self.mlp_q], 1)

            for layer_size in self.layer_sizes[1:]:
                output = slim.layers.fully_connected(output, num_outputs=layer_size, activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                         scale=1.0, mode="fan_avg", distribution="uniform", seed=self.seed))
            self.mlp_vector = output

        with tf.compat.v1.variable_scope("ncf", reuse=tf.compat.v1.AUTO_REUSE):
            if self.model_type == "gmf":
                output = slim.layers.fully_connected(self.gmf_vector, num_outputs=1, activation_fn=None, biases_initializer=None,
                                                     weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                         scale=1.0, mode="fan_avg", distribution="uniform", seed=self.seed))
                self.output = tf.sigmoid(output)
            elif self.model_type == "mlp":
                output = slim.layers.fully_connected(self.mlp_vector, num_outputs=1, activation_fn=None, biases_initializer=None,
                                                     weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                         scale=1.0, mode="fan_avg", distribution="uniform", seed=self.seed))
                self.output = tf.sigmoid(output)
            elif self.model_type == "neumf":
                self.ncf_vector = tf.concat([self.gmf_vector, self.mlp_vector], 1)
                output = slim.layers.fully_connected(self.ncf_vector, num_outputs=1, activation_fn=None, biases_initializer=None,
                                                     weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                         scale=1.0, mode="fan_avg", distribution="uniform", seed=self.seed))
                self.output = tf.sigmoid(output)

        with tf.compat.v1.variable_scope("loss", reuse=tf.compat.v1.AUTO_REUSE):
            self.loss = tf.compat.v1.losses.log_loss(self.labels, self.output)

        with tf.compat.v1.variable_scope("optimizer", reuse=tf.compat.v1.AUTO_REUSE):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, os.path.join(dir_name, MODEL_CHECKPOINT))

    def load(self, gmf_dir=None, mlp_dir=None, neumf_dir=None, alpha=0.5):
        if self.model_type == "gmf" and gmf_dir is not None:
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, os.path.join(gmf_dir, MODEL_CHECKPOINT))
        elif self.model_type == "mlp" and mlp_dir is not None:
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, os.path.join(mlp_dir, MODEL_CHECKPOINT))
        elif self.model_type == "neumf" and neumf_dir is not None:
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, os.path.join(neumf_dir, MODEL_CHECKPOINT))
        elif self.model_type == "neumf" and gmf_dir is not None and mlp_dir is not None:
            self._load_neumf(gmf_dir, mlp_dir, alpha)
        else:
            raise NotImplementedError

    def _load_neumf(self, gmf_dir, mlp_dir, alpha):
        variables = tf.compat.v1.global_variables()
        var_flow_restore = [val for val in variables if "gmf" in val.name and "ncf" not in val.name]
        saver = tf.compat.v1.train.Saver(var_flow_restore)
        saver.restore(self.sess, os.path.join(gmf_dir, MODEL_CHECKPOINT))

        variables = tf.compat.v1.global_variables()
        var_flow_restore = [val for val in variables if "mlp" in val.name and "ncf" not in val.name]
        saver = tf.compat.v1.train.Saver(var_flow_restore)
        saver.restore(self.sess, os.path.join(mlp_dir, MODEL_CHECKPOINT))

        variables = tf.compat.v1.global_variables()
        var_flow_param = [val for val in variables if "ncf" in val.name or "learning_rate" in val.name]

        self.sess.run(tf.compat.v1.variables_initializer(var_flow_param))

        gmf_vars = [var for var in variables if "gmf" in var.name and "ncf" not in var.name and "Adam" not in var.name]
        mlp_vars = [var for var in variables if "mlp" in var.name and "ncf" not in var.name and "Adam" not in var.name]
        ncf_vars = [var for var in variables if "ncf" in var.name and "Adam" not in var.name]

        assert len(gmf_vars) == len(mlp_vars)

        for gmf_var, mlp_var, ncf_var in zip(gmf_vars, mlp_vars, ncf_vars):
            self.sess.run(ncf_var.assign(tf.add(tf.multiply(alpha, gmf_var), tf.multiply(1 - alpha, mlp_var))))

    def fit(self, X, y, validation_data):
        for epoch in range(1, self.n_epochs + 1):
            t1 = time.time()
            loss_train = self._train_model(X, y)
            t2 = time.time()
            loss_val = self._eval_model(validation_data)
            if epoch % self.verbose == 0:
                print(f"Epoch {epoch} [{t2 - t1:.1f}s]: train_loss = {loss_train:.4f}, val_loss = {loss_val:.4f}")

    def _train_model(self, X, y):
        shuffle_indices = np.arange(len(y))
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices]
        y = y[shuffle_indices]

        n_batches = len(y) // self.batch_size + 1
        loss = 0
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(len(y), (i + 1) * self.batch_size)
            X_batch, y_batch = X[start:end], y[start:end]
            feed_dict = {self.user_input: X_batch[:, 0].reshape(-1, 1),
                         self.item_input: X_batch[:, 1].reshape(-1, 1),
                         self.labels: y_batch.reshape(-1, 1)}
            _, batch_loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            loss += batch_loss * len(y_batch)
        return loss / len(y)

    def _eval_model(self, validation_data):
        X_val, y_val = validation_data
        feed_dict = {self.user_input: X_val[:, 0].reshape(-1, 1),
                     self.item_input: X_val[:, 1].reshape(-1, 1),
                     self.labels: y_val.reshape(-1, 1)}
        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def predict(self, X):
        feed_dict = {self.user_input: X[:, 0].reshape(-1, 1),
                     self.item_input: X[:, 1].reshape(-1, 1)}
        predictions = self.sess.run(self.output, feed_dict=feed_dict)
        return predictions.flatten()

# Prepare training data
X_train = train_data[['user_index', 'item_index']].values
y_train = train_data['rating'].values

# Prepare validation data
X_val = test_data[['user_index', 'item_index']].values
y_val = test_data['rating'].values

# Instantiate and train the model
ncf = NCF(n_users, n_items, model_type="NeuMF", n_factors=8, layer_sizes=[16, 8, 4], n_epochs=10, batch_size=64, learning_rate=5e-3, verbose=1)
ncf.fit(X_train, y_train, validation_data=(X_val, y_val))

# Save the model
ncf.save("path_to_save_model")  # Replace with the actual path

# Generate recommendations
user_id = 1  # Replace with actual user_id
user_index = user_encoder.transform([user_id])[0]
item_indices = np.arange(n_items)
user_item_pairs = np.array([[user_index, item_index] for item_index in item_indices])

predictions = ncf.predict(user_item_pairs)
top_items = np.argsort(-predictions)[:10]

# Retrieve and display the top 10 recommended items
recommended_item_ids = item_encoder.inverse_transform(top_items)
print("Top 10 recommended items for user {}: {}".format(user_id, recommended_item_ids))
