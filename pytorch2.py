import re

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from scipy.sparse import hstack
from torch.utils.data import DataLoader, TensorDataset, Dataset

from experiment_runner import ExperimentRunner

BASIC_MODELS_GEN = {
    'basic': lambda: Model1(),
}

words_pattern = re.compile(r"(\b\w\w+\b)")
digit_pattern = re.compile(r"(\d+)")


def text_remove_alphanum(txt):
    txt = words_pattern.sub('A', txt)
    txt = digit_pattern.sub('D', txt)
    return txt


from torch import nn
from torchtext.vocab import build_vocab_from_iterator


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, max_sentece_len):
        super(TextClassificationModel, self).__init__()
        self.max_sentece_len = max_sentece_len
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.dropout = nn.Dropout()
        self.layer1 = nn.Linear(self.max_sentece_len * self.embed_dim, self.max_sentece_len)
        self.layer2 = nn.Linear(self.max_sentece_len, num_class)
        self.layer3 = nn.Linear(num_class, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for l in [self.layer1, self.layer2, self.layer3]:
            l.weight.data.uniform_(-initrange, initrange)
            l.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        embedded_compact = embedded.view(-1, self.max_sentece_len * self.embed_dim)
        relu = nn.ReLU()
        x = relu(self.layer1(self.dropout(embedded_compact)))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        return x

class TextClassificationModel2(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, max_sentece_len):
        super(TextClassificationModel2, self).__init__()
        self.max_sentece_len = max_sentece_len
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(1, 2, (13, 5), stride=5)
        self.fc = nn.Linear(self._calc_fc_input_size(), num_class)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded2 = embedded.unsqueeze(1)
        x = self.conv1(embedded2)
        x = self.relu(x)
        x = self.fc(x.view(text.size(0), -1))
        return x

    def _calc_fc_input_size(self):
        batch_size = 3
        embedded2 = torch.zeros(size=(batch_size, 1, self.max_sentece_len, self.embed_dim))
        x = self.conv1(embedded2)
        return x.view(batch_size, -1).size(1)


class DataSet:

    def __init__(self):
        pass


class NumpyDataset(Dataset):

    def __init__(self, numpys, max_sentence_size, vocab, tokenizer) -> None:
        self.numpys = numpys
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_sentence_size = max_sentence_size

    def __getitem__(self, index):
        return tuple(aray[index] for aray in self.numpys)

    def __len__(self):
        return len(self.numpys[0])

    def collate_batch(self, batch):
        text_list = []
        labels = torch.zeros(size=(len(batch),), dtype=torch.int64)
        text_tensor = torch.zeros(size=(len(batch), self.max_sentence_size), dtype=torch.int64)
        for idx, (txt, label_int) in enumerate(batch):
            processed_text = torch.tensor(self.vocab(self.tokenizer(txt)), dtype=torch.int64)
            text_list.append(processed_text)
            labels[idx] = label_int
            if len(processed_text) >= self.max_sentence_size:
                text_tensor[idx, :self.max_sentence_size] = processed_text[:self.max_sentence_size]
            else:
                repeats = self.max_sentence_size // len(processed_text)
                text_tensor[idx, :repeats*len(processed_text)] = processed_text.repeat(repeats)
                text_tensor[idx, repeats*len(processed_text):] = processed_text[:self.max_sentence_size-repeats*len(processed_text)]
        return text_tensor, labels.long()


import time


def train(model, optimizer, criterion, dataloader, device):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (text, label) in enumerate(dataloader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('{:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            text, label = text.to(device), label.to(device)
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, float(loss)


class Model1:

    def __init__(self):
        self.vectorizer_no_aplhanum = CountVectorizer(lowercase=False, token_pattern=r"[^\s]")
        self.model = SVC()

    def fit(self, train_x, train_y):
        tokenizer = self.vectorizer_no_aplhanum.build_tokenizer()
        train_x_no_aplhanum = np.vectorize(text_remove_alphanum)(train_x)
        max_sentence_size = [-1]

        def train_iterator():
            max_sentence_size_local = 0
            for txt in train_x_no_aplhanum:
                tokens = tokenizer(txt)
                max_sentence_size_local = max(max_sentence_size_local, len(tokens))
                yield tokens
            max_sentence_size[0] = max_sentence_size_local

        vocab = build_vocab_from_iterator(train_iterator(), specials=["<unk>"])
        max_sentence_size = max_sentence_size[0]
        max_sentence_size = min(max_sentence_size, 2000)
        vocab.set_default_index(vocab["<unk>"])
        x_vector = self.vectorizer_no_aplhanum.fit_transform(train_x_no_aplhanum)
        self.label_encoder = LabelEncoder()
        train_y_encoded = self.label_encoder.fit_transform(train_y)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = NumpyDataset([train_x_no_aplhanum, train_y_encoded], max_sentence_size, vocab, tokenizer)
        self.collate_fn = dataset.collate_batch
        dl = DataLoader(dataset, batch_size=70, shuffle=True, collate_fn=dataset.collate_batch)

        self.model = TextClassificationModel2(vocab_size=len(vocab), embed_dim=30, num_class=len(self.label_encoder.classes_),
                                             max_sentece_len=max_sentence_size).to(device)

        EPOCHS = 10
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train(model=self.model, optimizer=optimizer, criterion=criterion, dataloader=dl, device=device)
            accu_val, loss = evaluate(model=self.model, criterion=criterion, dataloader=dl, device=device)
            print(f"{accu_val = }")
            print(f"{loss = }")

    def predict(self, x):
        x_tensor, _ = self.collate_fn(list(zip(x, torch.zeros(size=(len(x),)))))
        self.model.eval()
        m = self.model.cpu()
        y = m(x_tensor)
        return self.label_encoder.inverse_transform(y.argmax(1).tolist())


def eval_basic_models(verbose):
    r = ExperimentRunner('./data.csv', verbose=verbose)
    r.train_eval_models(BASIC_MODELS_GEN, './stats-model1-models')


if __name__ == '__main__':
    # TODO: dotenv
    eval_basic_models(verbose=True)
