import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Fragments
from Chemprompt.config.custom_model import get_model_repo

class LLMModel:
    def __init__(self, model_repo, dtype="full", device = "cuda:0"):
        model_info = get_model_repo(model_repo)
        repo = model_info['repo']
        name = model_info['name']
        model_path = f"{repo}/{name}"

        self.name = name
        self.dtype = torch.float16 if dtype == "half" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype
        ).to(device)

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_embeddings(self, smiles_list):
        embeddings = []
        device = self.model.device

        for smiles in tqdm(smiles_list, desc="Processing SMILES with Properties"):
            input_text = self.create_input_text(smiles)
            if input_text is None:
                print(f"Skipping SMILES: {smiles} due to missing properties.")
                continue

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                # truncation=True,
                # max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            embeddings.append(embedding)

        return np.array(embeddings)
    
    def create_input_text(self, smiles):
        prompt = f"SMILES: {smiles}"
        # print(prompt)
        return prompt