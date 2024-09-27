import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as transforms
from comet_ml import Experiment
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

train_csv = '/teamspace/studios/this_studio/ASR-with-Speech-Sentiment-and-Text-Summarizer/src/ASR-with-Speech-Sentiment-Analysis-Text-Summarizer/Speech_Sentiment_Analysis/scripts/output/train.csv'
test_csv = '/teamspace/studios/this_studio/ASR-with-Speech-Sentiment-and-Text-Summarizer/src/ASR-with-Speech-Sentiment-Analysis-Text-Summarizer/Speech_Sentiment_Analysis/scripts/output/test.csv'

train_df = pd.read_csv(train_csv, names=["Path", "Emotions"])
test_df = pd.read_csv(test_csv, names=["Path", "Emotions"])

train_df = train_df[train_df["Emotions"] != "Emotions"]
test_df = test_df[test_df["Emotions"] != "Emotions"]

emotion_to_idx = {label: idx for idx,
                  label in enumerate(train_df['Emotions'].unique())}


class SERDataset(Dataset):
    def __init__(self, df, emotion_to_idx, transform=None, max_len=16000):
        self.df = df
        self.transform = transform
        self.emotion_to_idx = emotion_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx, 0]
        emotion = self.df.iloc[idx, 1]
        emotion_idx = self.emotion_to_idx[emotion]

        waveform, sample_rate = torchaudio.load(audio_path)

        # To fix dual channel audio problem
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_spec = self.transform(waveform)

        if mel_spec.shape[2] > self.max_len:
            mel_spec = mel_spec[:, :, :self.max_len]
        else:
            padding = self.max_len - mel_spec.shape[2]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))

        return mel_spec, emotion_idx


batch_size = 32
learning_rate = 1e-4
num_epochs = 5
num_classes = len(emotion_to_idx)
sequence_length = 100
max_len = 16000

mel_transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

train_dataset = SERDataset(train_df, emotion_to_idx,
                           transform=mel_transform, max_len=max_len)
test_dataset = SERDataset(test_df, emotion_to_idx,
                          transform=mel_transform, max_len=max_len)


def collate_fn(batch):
    mel_specs, labels = zip(*batch)
    mel_specs = torch.stack(mel_specs)
    labels = torch.tensor(labels)
    return mel_specs, labels


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=collate_fn)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores = self.Va(torch.tanh(self.Wa(lstm_out) + self.Ua(lstm_out)))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context


class CNN_LSTM_Attention(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_Attention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(input_size=32 * 16, hidden_size=128,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size=128 * 2)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):

        cnn_out = self.cnn(x)

        batch_size, num_channels, n_mels, seq_len = cnn_out.shape
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous()
        cnn_out = cnn_out.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(cnn_out)

        context = self.attention(lstm_out)

        output = self.fc(context)
        return output


model = CNN_LSTM_Attention(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORKSPACE")
    )

    experiment.log_parameters({
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": train_loader.batch_size,
        "num_epochs": num_epochs,
        "model": model.__class__.__name__
    })

    best_accuracy = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            step_loss = loss.item()
            step_accuracy = 100. * \
                predicted.eq(targets).sum().item() / targets.size(0)

            experiment.log_metric(
                "train_step_loss", step_loss, step=global_step)
            experiment.log_metric("train_step_accuracy",
                                  step_accuracy, step=global_step)

            train_pbar.set_postfix({'Loss': train_loss / (batch_idx + 1),
                                    'Acc': 100. * train_correct / train_total})

            global_step += 1

        train_accuracy = 100. * train_correct / train_total
        experiment.log_metric("train_epoch_loss",
                              train_loss / len(train_loader), step=epoch)
        experiment.log_metric("train_epoch_accuracy",
                              train_accuracy, step=epoch)

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            test_pbar = tqdm(
                test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch_idx, (inputs, targets) in enumerate(test_pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                step_loss = loss.item()
                step_accuracy = 100. * \
                    predicted.eq(targets).sum().item() / targets.size(0)

                experiment.log_metric(
                    "test_step_loss", step_loss, step=global_step)
                experiment.log_metric(
                    "test_step_accuracy", step_accuracy, step=global_step)

                test_pbar.set_postfix({'Loss': test_loss / (batch_idx + 1),
                                       'Acc': 100. * test_correct / test_total})

        test_accuracy = 100. * test_correct / test_total
        experiment.log_metric(
            "test_epoch_loss", test_loss / len(test_loader), step=epoch)
        experiment.log_metric("test_epoch_accuracy", test_accuracy, step=epoch)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(
            f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
        print(
            f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

            experiment.log_model("best_model", 'best_model.pth')

        print()
    experiment.end()


train_and_evaluate(model, train_loader, test_loader,
                   criterion, optimizer, num_epochs, device)
