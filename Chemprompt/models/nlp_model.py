import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List

class ModelTrainer:
    def __init__(self, model_type: str, task_type: str, x: List[str], y: List, data_name, num_folds: int = 5, batch_size: int = 128, lr=1e-4, device="cuda:0", num_epochs: int = 100, max_length: int = 2048):
        self.model_type = model_type.lower()
        self.task_type = task_type.lower()
        self.x = x
        self.y = np.array(y)
        self.data_name = data_name
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.max_length = max_length
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.results_path = f"./result/{self.model_type}"
        os.makedirs(self.results_path, exist_ok=True)

        self.metrics = (
            ["F1-micro", "F1-macro"] 
            if self.task_type == "classification" 
            else ["RMSE", "R2", "PCC", "Spearman"]
        )

    def tokenize_smiles(self, smiles_list: List[str]):
        if self.model_type == "bert":
            return self.tokenizer(smiles_list, padding="longest", truncation=True, max_length=self.max_length, return_tensors="pt")
        else:
            return self.tokenizer(smiles_list, padding="longest", truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=False)

    def get_model(self):
        vocab_size = self.tokenizer.vocab_size
        if self.model_type == "bert":
            return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.y.shape[1]).to(self.device)
        elif self.model_type == "lstm":
            return LSTMClassifier(vocab_size, num_classes=self.y.shape[1]).to(self.device)
        elif self.model_type == "rnn":
            return RNNClassifier(vocab_size, num_classes=self.y.shape[1]).to(self.device)
        else:
            raise ValueError("Unsupported model type.")

    def collate_fn(self, batch):
        input_ids, attention_masks, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        lengths = (input_ids != 0).sum(dim=1)
    
        return input_ids, attention_masks, labels, lengths

    def train_epoch(self, model, dataloader, optimizer, loss_fn):
        model.train()
        cnt = 0
        total_loss = 0
        for input_ids, attention_mask, labels, lengths in dataloader:
            input_ids, attention_mask, labels, lengths = (
                input_ids.to(self.device), 
                attention_mask.to(self.device), 
                labels.to(self.device), 
                lengths.to(self.device)
            )
            optimizer.zero_grad()
            
            # Extract logits when using BERT (SequenceClassifierOutput)
            if self.model_type == "bert":
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits  # Only use logits
            else:
                logits = model(input_ids, lengths)
    
            if self.task_type == "classification":
                loss = loss_fn(logits, labels.float())  # BCEWithLogitsLoss requires float type labels
            else:  # For regression: Exclude padding tokens when calculating MSELoss
                mask = (input_ids != self.tokenizer.pad_token_id).float().unsqueeze(-1)  # Padding mask (0 = padding)
                masked_loss = (loss_fn(logits, labels) * mask).sum() / mask.sum()  # Compute MSELoss without padding
                loss = masked_loss
    
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        return total_loss / len(dataloader)


    def train_and_evaluate(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        all_predictions = []
        metric_results = []
    
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.x), start=1):
            x_train, x_test = [self.x[i] for i in train_idx], [self.x[i] for i in test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
    
            train_encodings = self.tokenize_smiles(x_train)
            test_encodings = self.tokenize_smiles(x_test)
    
            train_labels = torch.tensor(y_train, dtype=torch.float)
            test_labels = torch.tensor(y_test, dtype=torch.float)
    
            train_dataset = list(zip(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels))
            test_dataset = list(zip(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels))
    
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
            model = self.get_model()
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-2)
    
            loss_fn = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()
    
            for epoch in range(1, self.num_epochs + 1):
                avg_loss = self.train_epoch(model, train_loader, optimizer, loss_fn)
                if epoch % (self.num_epochs // 20) == 0 or epoch == self.num_epochs:
                    print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.6f}")
    
            scores, y_pred = self.evaluate(model, test_loader, y_test, y_train)
    
            metric_results.append({"Fold": fold, **scores})
    
            for local_idx, (idx, pred) in enumerate(zip(test_idx, y_pred)):
                all_predictions.append({
                    "Fold": fold,
                    "SMILES": self.x[idx],
                    "GroundTruth": y_test[local_idx].tolist(),
                    "Prediction": pred.tolist()
                })
    
        self.save_predictions(all_predictions)
        self.save_combined_metrics(metric_results)


    def evaluate(self, model, dataloader, y_true, y_train=None):
        model.eval()
        y_pred = []
        with torch.no_grad():
            for input_ids, attention_mask, _, lengths in dataloader:
                input_ids, attention_mask, lengths = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    lengths.to(self.device),
                )
    
                outputs = model(input_ids, attention_mask) if self.model_type == "bert" else model(input_ids, lengths)
                logits = outputs.logits if self.model_type == "bert" else outputs
    
                y_pred.extend(logits.cpu().numpy())
    
        y_pred = np.array(y_pred)
    
        if self.task_type == "classification":
            from scipy.special import expit
            y_pred_binary = (expit(y_pred) > 0.5).astype(int)
            scores = {
                "F1-micro": f1_score(y_true, y_pred_binary, average="micro"),
                "F1-macro": f1_score(y_true, y_pred_binary, average="macro"),
            }
            return scores, y_pred_binary
    
        else:
            mu = float(np.mean(y_train))
            sigma = float(np.std(y_train))
    
            y_test_vec = y_true.ravel()
            y_pred_vec = y_pred.ravel()
    
            if sigma > 0:
                y_test_z = (y_test_vec - mu) / sigma
                y_pred_z = (y_pred_vec - mu) / sigma
            else:
                y_test_z = y_test_vec
                y_pred_z = y_pred_vec
    
            rmse_value = np.sqrt(mean_squared_error(y_test_z, y_pred_z))
            r2_value = r2_score(y_test_z, y_pred_z)
            pcc_value, _ = pearsonr(y_test_z, y_pred_z)
            spearman_value, _ = spearmanr(y_test_z, y_pred_z)
    
            print(f"[Fold Regression Metrics] RMSE={rmse_value:.3f}, R2={r2_value:.3f}, PCC={pcc_value:.3f}, Spearman={spearman_value:.3f}")
    
            scores = {
                "RMSE": rmse_value,
                "R2": r2_value,
                "PCC": pcc_value,
                "Spearman": spearman_value
            }
            return scores, y_pred

    def save_predictions(self, all_predictions):
        df = pd.DataFrame(all_predictions, columns=["Fold", "SMILES", "GroundTruth", "Prediction"])
        predictions_path = os.path.join(self.results_path, f"predictions_{self.data_name}.csv")
        df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

    def save_combined_metrics(self, metric_results):
        combined_results = {
            "Metric": [],
            "Mean": [],
            "Standard Deviation": []
        }
    
        # Convert metric_results into a DataFrame
        df = pd.DataFrame(metric_results)
    
        # Debugging: Print available metrics
        print("Metrics List:", self.metrics)
        print("DataFrame Columns:", df.columns)
    
        for metric in self.metrics:
            if metric in df.columns:
                mean_value = df[metric].mean()
                std_value = df[metric].std()
            else:
                print(f"Warning: {metric} not found in DataFrame!")  # Debugging
                mean_value = None
                std_value = None
    
            combined_results["Metric"].append(metric)
            combined_results["Mean"].append(mean_value)
            combined_results["Standard Deviation"].append(std_value)
    
        # Convert to DataFrame for saving
        results_df = pd.DataFrame(combined_results)
    
        # Define save path
        metrics_path = os.path.join(self.results_path, f"metrics_{self.data_name}.csv")
    
        # Save to CSV
        results_df.to_csv(metrics_path, index=False)
    
        print(f"Metrics saved to {metrics_path}")


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=1024, num_layers=4, num_classes=12, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, 
                          nonlinearity="tanh", bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # bidirectional 
        # self.softmax = nn.Softmax(dim=1)  # softmax

    def forward(self, input_ids, lengths):
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        out, hidden = self.rnn(x)
        out, _ = pad_packed_sequence(out, batch_first=True)

        last_outputs = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim * 2)

        logits = self.fc(last_outputs)  # (batch, num_classes)
        return logits

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=1024, num_layers=4, num_classes=12, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=True,  batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        # self.softmax = nn.Softmax(dim=1)  # add Softmax

    def forward(self, input_ids, lengths):
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        out, (hn, _) = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first=True)

        # Select the last valid timestep of each sequence (excluding padded parts)
        idx = (lengths - 1).unsqueeze(1).expand(-1, out.size(2))  # (batch, hidden_dim)
        #last_outputs = out.gather(1, idx.unsqueeze(1)).squeeze(1)  # (batch, hidden_dim)
        last_outputs = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # (batch, hidden_dim * 2)

        logits = self.fc(last_outputs)

        return logits  # (batch, num_classes)

class BertRegression(nn.Module):
    def __init__(self):
        super(BertRegression, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze()