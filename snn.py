import argparse
import os
import dataset
import network as net
import numpy as np
from sklearn.metrics import accuracy_score

def run(path):
    audio_files = os.listdir(path)
    # audio_files = audio_files[:600]

    X_train, X_test, y_train, y_test = dataset.load(audio_files, path, test_size=0.2)

    
    print("Features shape: ", X_train.shape)
    print("Labels shape: ",y_train.shape)

    print("X_train: ", X_train)
    print("labels: ", y_train)



    #####################
    # Convert labels to one-hot encoding if necessary
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))

    ############

    T = 1000
    dt = 0.1

    input_dim = 200
    output_dim = 10
    vdecay = 0.5
    vth = 0.5
    snn_timesteps = 20

    input_2_output_weights = np.random.rand(output_dim, input_dim)

    print("Initial weights: ", input_2_output_weights)

    # snn using the arguments defined above
    snn_network = net.SNN(input_2_output_weights, input_dim, output_dim, vdecay, vth, snn_timesteps)

    spike_encoded_inputs = X_train

    train_data = list(zip(spike_encoded_inputs, y_train_encoded))

    print("train data: ", train_data)

    # Train network
    # net.Learning.hebbian(snn_network, train_data, lr=1e-5, epochs=10)

    # object of STDP class with appropriate arguments
    A_plus = 0.04
    A_minus = 0.003
    tau_plus = 15
    tau_minus = 20
    lr = 1e-4
    epochs = 50

    stdp = net.STDP(snn_network, A_plus, A_minus, tau_plus, tau_minus, lr, snn_timesteps, epochs)
    stdp.train(train_data)

    print("Training completed successfully!")


    # Test the network
    spike_encoded_tests = X_test
    test_outputs = np.array([snn_network(spike_input) for spike_input in spike_encoded_tests])

    # Assuming the output is in a format that matches y_test_encoded, use argmax to convert from probabilities
    predicted_labels = np.argmax(test_outputs, axis=1)

    y_test_encoded = encoder.fit_transform(y_test.reshape(-1, 1))
    true_labels = np.argmax(y_test_encoded, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Test accuracy: {accuracy:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="./.data/recordings",
        type=str,
        help="Path to the downloaded data files",
    )
    args = parser.parse_args()

    run(args.path)
