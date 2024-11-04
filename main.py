import torch
from torch.utils.data import DataLoader
from src.data.make_dataset import load_data, prepare_data, build_embedding_dictionary, convert_to_embeddings
from src.model.LSTMNet import LSTMNet
from src.train import train_model, FastTextDataset
from cfg.training_config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    NUM_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
    LEARNING_RATE,
    NUM_EPOCHS,
)

df = load_data()
X_train, X_test, y_train, y_test, corpus = prepare_data(df)

emb_dict = build_embedding_dictionary(corpus)
X_train_emb = convert_to_embeddings(X_train, emb_dict)
X_test_emb = convert_to_embeddings(X_test, emb_dict)

train_dataset = FastTextDataset(X_train_emb, y_train)
test_dataset = FastTextDataset(X_test_emb, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LSTMNet(
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=NUM_LAYERS,
    bidirectional=BIDIRECTIONAL,
    dropout=DROPOUT,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss().to(device)

train_loss_hist, test_loss_hist = train_model(
    model,
    optimizer,
    criterion,
    train_dataloader,
    test_dataloader,
    NUM_EPOCHS,
    device
)
