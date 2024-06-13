
# Wildlife Monitor

## Objective
In this project, we developed a system to monitor wildlife using images from the Oxford IIIT Pet dataset. We used a convolutional neural network (CNN) to classify animals in the images and analyzed the data to gain insights into wildlife behavior and population trends.

## Dataset
The Oxford IIIT Pet dataset can be directly imported using TensorFlow Datasets.

## Model
We used a convolutional neural network (CNN) for image classification with the following architecture:
- Conv2D layer with 32 filters and (3, 3) kernel size
- MaxPooling2D layer with (2, 2) pool size
- Conv2D layer with 64 filters and (3, 3) kernel size
- MaxPooling2D layer with (2, 2) pool size
- Conv2D layer with 128 filters and (3, 3) kernel size
- MaxPooling2D layer with (2, 2) pool size
- Flatten layer
- Dense layer with 512 units and ReLU activation
- Dense output layer with softmax activation

## Installation
To run this project, you need to have the following libraries installed:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- tensorflow_datasets
- scikit-learn

You can install the required libraries using pip:
```sh
pip install numpy pandas matplotlib seaborn tensorflow tensorflow_datasets scikit-learn
```

## Usage
Run the Jupyter Notebook wildlife_monitoring.ipynb to train and evaluate the model.

## License
This project is licensed under the MIT License.

----------------------------------------------------------------

Feel free to suggest any kind of improvements.