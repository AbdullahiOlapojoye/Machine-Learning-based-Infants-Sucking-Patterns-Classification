{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AbdullahiOlapojoye/Machine-Learning-based-Infants-Sucking-Patterns-Classification/blob/main/CNN_final.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "v5qy9y2dsZMP"
      },
      "outputs": [],
      "source": [
        "#Libraries\n",
        "import os\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from keras.models import Sequential\n",
        "from keras.layers import SimpleRNN, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D\n",
        "from keras.optimizers import Adam\n",
        "from keras.models import load_model\n",
        "from keras.callbacks import ModelCheckpoint\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# For the Visuals\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib as mpl\n",
        "from matplotlib import cm\n",
        "import matplotlib.patches as patches\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.image as mpimg\n",
        "from matplotlib.ticker import MaxNLocator\n",
        "from matplotlib.offsetbox import AnnotationBbox, OffsetImage\n",
        "from matplotlib.colors import ListedColormap, LinearSegmentedColormap\n",
        "from matplotlib.patches import Rectangle\n",
        "from IPython.display import display_html\n",
        "plt.rcParams.update({'font.size': 16})\n",
        "import plotly.graph_objects as go\n",
        "#performance\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.metrics import classification_report\n",
        "from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_fscore_support\n",
        "from sklearn.metrics import roc_auc_score,auc,f1_score\n",
        "from sklearn.metrics import precision_recall_curve,roc_curve"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **PREPARE DATA**"
      ],
      "metadata": {
        "id": "UWIts31Y5inS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to read and process CSV files\n",
        "def read_and_process_csv(file_path):\n",
        "    df = pd.read_csv(file_path)\n",
        "    data = df[['F1', 'F2']].values\n",
        "    return data\n",
        "\n",
        "# Function to read labels from the \"targets.csv\" file\n",
        "def read_labels_from_targets(targets_file_path):\n",
        "    df_targets = pd.read_csv(targets_file_path)\n",
        "    return df_targets.values\n",
        "\n",
        "# Function to generate data and labels from a list of CSV files\n",
        "def generate_data_from_csv(file_list, targets_file_path):\n",
        "    data_list = []\n",
        "\n",
        "    for file_path in file_list:\n",
        "        time_series_data = read_and_process_csv(file_path)\n",
        "        data_list.append(time_series_data)\n",
        "\n",
        "    labels = read_labels_from_targets(targets_file_path)\n",
        "\n",
        "    return np.array(data_list), np.array(labels)\n",
        "\n",
        "# Specify the path to your CSV files\n",
        "path1 = \"/content/drive/MyDrive/BMEN 6367/SENSOR_DATA_DL_m/Features/Subject_\"\n",
        "targets_file_path = \"/content/drive/MyDrive/BMEN 6367/SENSOR_DATA_DL_V2/Target/Target.csv\"\n",
        "#files\n",
        "csv_files = []\n",
        "for i in range(1,41):\n",
        "    file_path = path1 + str(i) + '.csv'\n",
        "    print(file_path)\n",
        "    csv_files.append(file_path)\n",
        "\n",
        "# Generate data and labels\n",
        "X, y = generate_data_from_csv(csv_files, targets_file_path)\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)"
      ],
      "metadata": {
        "id": "PAEEMFzQsi5m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **MODELLING WITH 1D-CNN**"
      ],
      "metadata": {
        "id": "7xypRUMC5qTa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import LSTM, Dense\n",
        "# Set hyperparameters\n",
        "input_shape = X_train.shape[1:]\n",
        "num_classes = len(np.unique(y_train))\n",
        "num_filters = 16\n",
        "kernel_size = 3\n",
        "pool_size = 2\n",
        "dropout_rate = 0.5\n",
        "batch_size = 64\n",
        "epochs = 20\n",
        "\n",
        "# Build the 1D CNN model_1D\n",
        "model_1D = Sequential()\n",
        "model_1D.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(1400, 2)))\n",
        "model_1D.add(MaxPooling1D(pool_size=pool_size))\n",
        "model_1D.add(Conv1D(filters=num_filters * 4, kernel_size=kernel_size, activation='relu'))\n",
        "model_1D.add(MaxPooling1D(pool_size=pool_size))\n",
        "model_1D.add(Conv1D(filters=num_filters * 2, kernel_size=kernel_size, activation='relu'))\n",
        "model_1D.add(MaxPooling1D(pool_size=pool_size))\n",
        "model_1D.add(Flatten())\n",
        "model_1D.add(Dense(1, activation='sigmoid'))\n",
        "\n",
        "# Compile the model\n",
        "adam = Adam(learning_rate=0.001)\n",
        "model_1D.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "# Define 5-fold cross-validation\n",
        "kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "\n",
        "# Perform cross-validation\n",
        "for train_index, val_index in kfold.split(X_train, y_train):\n",
        "    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]\n",
        "    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]\n",
        "\n",
        "    # Train the model on the current fold\n",
        "    chk = ModelCheckpoint('best_model_cnn.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)\n",
        "    model_1D.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, callbacks=[chk], validation_data=(X_val_fold, y_val_fold))\n",
        "\n",
        "    # Evaluate the model on the validation set of the current fold\n",
        "    results = model_1D.evaluate(X_val_fold, y_val_fold)\n",
        "    print(f'Fold Accuracy: {results[1]}')\n",
        "\n",
        "# After the loop, you can evaluate the overall performance of your model on the test set\n",
        "test_results = model_1D.evaluate(X_test, y_test)\n",
        "print(f'Test Accuracy: {test_results[1]}')"
      ],
      "metadata": {
        "id": "gxT8q1-7sptm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Performance Evaluation**"
      ],
      "metadata": {
        "id": "5zp298oF50Vg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = model_1D.predict(X_test).flatten()\n",
        "y_pred_CNN = (y_pred > 0.5).astype(int)\n",
        "# Create confusion matrix\n",
        "cm = confusion_matrix(y_test, y_pred_CNN)\n",
        "\n",
        "# Plot confusion matrix using seaborn\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Set1', xticklabels=['Healthy', 'Unhealthy'], yticklabels=['Healthy', 'Unhealthy'])\n",
        "plt.title('Confusion Matrix(1D-CNN)')\n",
        "plt.xlabel('Predicted Label')\n",
        "plt.ylabel('True Label')\n",
        "plt.show()\n",
        "\n",
        "# Display classification report\n",
        "print(classification_report(y_test, y_pred_CNN))"
      ],
      "metadata": {
        "id": "za8zS7-yNUbC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Compute ROC curve and ROC area for each class\n",
        "y_pred_CNN = model_1D.predict(X_test).flatten()\n",
        "fpr, tpr, _ = roc_curve(y_test, y_pred_CNN)\n",
        "roc_auc = auc(fpr, tpr)\n",
        "\n",
        "# Plot ROC curve\n",
        "plt.figure(figsize=(8, 6))\n",
        "plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))\n",
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate')\n",
        "plt.title('Receiver Operating Characteristic (ROC) Curve')\n",
        "plt.legend(loc='lower right')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "MwExQjXtPAMa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "# Save the model to a file\n",
        "model_cnn = 'trained_cnn_model.joblib'\n",
        "joblib.dump(model_1D, model_cnn)\n",
        "print(f\"Model saved to {model_cnn}\")\n",
        "\n",
        "# Load the saved model\n",
        "loaded_model = joblib.load('trained_cnn_model.joblib')\n"
      ],
      "metadata": {
        "id": "jKW-_lEwmszk"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}