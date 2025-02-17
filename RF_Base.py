import time
import random

import joblib
from matplotlib import pyplot as plt
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Create the Sacred experiment
ex = Experiment(f'RF_experiment_{int(time.time())}')
ex.observers.append(MongoObserver(url='mongodb://127.0.0.1:27017', db_name='my_database'))

# Add FileStorageObserver to log experiment results in a folder
ex.observers.append(FileStorageObserver('my_experiment_logs'))


@ex.config
def config():
    n_estimators = 10
    criterion = 'gini'
    max_depth = 5
    dataset = "FashionMNIST"
    random_state = 42

class RandomForestBase:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.n_estimators = config["n_estimators"]
        self.criterion = config["criterion"]
        self.max_depth = config["max_depth"]
        self.dataset = config["dataset"]
        self.random_state = config["random_state"]

    def run(self):
        # Define a transform to normalize the data
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to a vector
        ])

        # Load the FashionMNIST dataset using torchvision
        train_data = datasets.FashionMNIST(root='../Neural-Network-Framework/F_MNIST_data', train=True, download=False, transform=transform)
        test_data = datasets.FashionMNIST(root='../Neural-Network-Framework/F_MNIST_data', train=False, download=False, transform=transform)

        # Convert the datasets into DataLoader objects
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        # Load data from the DataLoader
        X_train, self.y_train = next(iter(train_loader))
        X_test, self.y_test = next(iter(test_loader))

        # Convert data to numpy arrays and flatten
        X_train = X_train.numpy()  # Convert to numpy array
        X_test = X_test.numpy()

        # Standardize the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)

        # Create and train the RandomForest model
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Log accuracy to Sacred
        ex.log_scalar('accuracy', accuracy)

        print(f'Accuracy: {accuracy:.4f}')

    @ex.capture
    def save_model(self):
            """Saves the model."""
            filename = f"models/{self.config['dataset']}_model_{int(time.time())}.pth"
            joblib.dump(self.model, f"{filename}")
            ex.add_artifact(filename, content_type="application/octet-stream")

    def load_model(self, filename="model.pth"):
            """Loads the model and optimizer state from a file."""
            loaded_rf = joblib.load(f"{filename}")
            return self.model

    def predict_random_image(self, amount):
        collected_image = []
        collected_label = []
        collected_prediction = []
        for i in range(0, amount):
            # Predict a random image from the test set
            random_idx = random.randint(0, len(self.X_test) - 1)
            random_image = self.X_test[random_idx]
            true_label = self.y_test[random_idx]

            # Predict the class of the random image
            predicted_label = self.model.predict([random_image])[0]
            #collection of predictions
            # TODO: Display multiple images at once to check where possible mistakes are.
            """            collected_image.append(random_image)
            collected_label.append(true_label)
            collected_prediction.append(predicted_label)

            for i,j in enumerate(collected_image):
                collected_image[i] = collected_image[i].reshape(28,28)

            fig = plt.figure(figsize=(8, 8))
            columns = 4
            rows = 5
            for i in range(1, len(collected_image)):
                fig.add_subplot(rows, columns, i)
                plt.imshow(collected_image[i])
            plt.show()"""
            # Visualize the image
            random_image_reshaped = random_image.reshape(28, 28)  # Reshape back to 28x28 for visualization
            plt.imshow(random_image_reshaped, cmap='gray')
            plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
            plt.show()

@ex.automain
def RFCall(_config):
    rf = RandomForestBase(_config)
    rf.run()
    rf.save_model()
    rf.predict_random_image(2)
