# RNN Tweets Binary Classification

## Project Description

This project implements a Recurrent Neural Network (RNN) for the binary classification of tweets. The objective is to classify tweets into two categories, enabling sentiment analysis and understanding public opinion on various topics. This model leverages various embedding techniques and hyperparameters to optimize classification performance.

### Features

- Utilizes RNN and LSTM architectures for sequence modeling.
- Supports different word embedding techniques such as Word2Vec, FastText, and GloVe.
- Configurable training parameters including batch size, learning rate, and number of epochs.
- Logs training progress and results using the `loguru` library.
- Data preprocessing and visualization capabilities.

## Requirements

To run this project, ensure you have the following dependencies installed. You can install them using pip or poetry.

### Using pip

```bash
pip install -r requirements.txt
```
### Using pip
```bash
poetry install
```
### Required Packages
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch
- torchvision
- torchaudio
- opencv-python
- datasets
- nltk
- gensim

### Instructions to Run

1. Clone the Repository
2. Navigate to the Project Directory
3. Install Dependencies
4. Prepare your Data
5. Configure Training Parameters
6. Run the Training

Execute the main script to start training model:
```bash
python main.py
```
