import torch

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 50
NUM_EPOCHS = 10

MODEL_TYPE = "LSTM"  
EMBEDDING_TYPE = "FastText"  

HIDDEN_DIM = 64
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
OUTPUT_DIM = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
