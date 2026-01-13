from __future__ import annotations
import os
import ast
import inspect
from collections import OrderedDict
from typing import List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, rdMolDescriptors,
    Fragments, QED,
)

from Chemprompt.config.custom_model import get_model_repo

class LLMModel:
    """LLM + RDKit descriptor text prompts → Extract embeddings"""

    def __init__(self, model_repo: str, dtype: str = "half", device: str = "cuda"):
        # model load
        model_info = get_model_repo(model_repo)
        repo, name = model_info["repo"], model_info["name"]
        ckpt = f"{repo}/{name}"

        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "half" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, legacy=False)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=self.dtype).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Descriptor filtering is performed lazily
        self._descriptor_funcs: OrderedDict[str, callable] = OrderedDict()
        self.property_names: list[str] = []
        self.property_funcs: list[callable] = []

    def _filter_descriptors(self, smiles_list: list[str]):
        """Collect all RDKit descriptors, compute them for given SMILES,
        and filter out low-variability descriptors (std <= 1 and unique <= 10)."""
    
        def _safe_len(x):
            """Safely return length, or None if not applicable."""
            try:
                return len(x)
            except:
                return None
    
        def _count_if_iterable(val):
            """Return len(val) if it's iterable, otherwise return val itself."""
            try:
                if isinstance(val, (list, tuple)):
                    return len(val)
            except:
                pass
            return val
    
        # Collect all RDKit descriptor functions
        d_all: OrderedDict[str, callable] = OrderedDict()
    
        # (1) Built-in descriptors
        for name, fn in Descriptors.descList:
            d_all[name] = fn
    
        # (2) Crippen descriptors
        for name in ("MolLogP", "MolMR"):
            d_all[name] = getattr(Crippen, name)
    
        # (3) Lipinski descriptors (convert list/tuple outputs to length)
        for name, func in inspect.getmembers(Lipinski, inspect.isfunction):
            def make_safe_func(f):
                return lambda m, _f=f: _count_if_iterable(_f(m))
            d_all[f"Lipinski_{name}"] = make_safe_func(func)
    
        # (4) rdMolDescriptors (exclude fingerprint-like functions)
        for name, fn in inspect.getmembers(rdMolDescriptors, inspect.isfunction):
            if "fingerprint" in name.lower():
                continue
            if len(inspect.signature(fn).parameters) == 1:
                d_all[f"rdMol_{name}"] = fn
    
        # (5) Fragment-based descriptors
        for name, fn in inspect.getmembers(Fragments, inspect.isfunction):
            d_all[name] = fn
    
        # (6) QED (single scalar)
        d_all["QED"] = QED.qed
    
        # --- Compute descriptor values for all SMILES ---
        names, funcs = list(d_all.keys()), list(d_all.values())
        rows = []
        for smi in tqdm(smiles_list, desc="Filtering descriptors"):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES -> fill row with None
                rows.append([None] * len(funcs))
                continue
            row = []
            for fn in funcs:
                try:
                    row.append(fn(mol))
                except:
                    row.append(None)
            rows.append(row)
    
        df = pd.DataFrame(rows, columns=names)
    
        # --- Calculate descriptor statistics ---
        df_stats = pd.DataFrame({
            "name": df.columns,
            "std": df.std(),
            "unique": df.nunique()
        })
    
        # --- Filter descriptors by variability ---
        selected_names = df_stats.query("std > 1 or unique > 10")["name"]
    
        # --- Keep only selected descriptors ---
        self._descriptor_funcs = OrderedDict(
            (n, d_all[n]) for n in selected_names if n in d_all
        )
    
        # Replace underscores for better readability
        self.property_names = [name.replace("_", " ") for name in self._descriptor_funcs.keys()]
    
        # --- Store callable functions (fixed closure binding) ---
        self.property_funcs = [lambda m, f=fn: f(m) for fn in self._descriptor_funcs.values()]

    
    def _calculate_properties(self, smiles: str, flag: Sequence[bool]):
        if len(flag) != len(self.property_names):
            raise ValueError("flag length mismatch with descriptor count")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * sum(flag)

        selected = [fn for keep, fn in zip(flag, self.property_funcs) if keep]
        out = []
        for fn in selected:
            try:
                out.append(fn(mol))
            except Exception:
                out.append(None)
        return out

    def _create_prompt(self, smiles: str, flag: Sequence[bool]):
        if not any(flag):
            return f"SMILES: {smiles}"

        props = self._calculate_properties(smiles, flag)
        names = [n for keep, n in zip(flag, self.property_names) if keep]
        parts = [f"{n}: {v:.5f}" if isinstance(v, (float, int)) else f"{n}: {v}"
                 for n, v in zip(names, props) if v is not None]

        prompt = f"SMILES: {smiles} | " + " | ".join(parts).replace('_', ' ') if parts else f"SMILES: {smiles}"
        # print(prompt)
        return prompt

    def get_embeddings(self, smiles_list: Sequence[str], flag: Sequence[bool] = None):
        # Perform descriptor filtering on the first call
        if not self.property_names:
            self._filter_descriptors(smiles_list)

        if flag is None:
            flag = [True] * len(self.property_names)
        if len(flag) != len(self.property_names):
            print(len(self.property_names))
            raise ValueError("flag length mismatch with descriptor count")

        vecs: list[np.ndarray] = []
        for smi in tqdm(smiles_list, desc="Embedding SMILES"):
            prompt = self._create_prompt(smi, flag)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device) # truncation=True
            with torch.no_grad():
                outputs = self.model(**inputs)
            vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            vecs.append(vec)
        return np.asarray(vecs)

    def get_embeddings_with_save(
        self,
        smiles_list: Sequence[str],
        flag: Sequence[bool],
        save_dir: str = "./embedding"
    ):
        # Perform descriptor filtering on the first call
        if not self.property_names:
            self._filter_descriptors(smiles_list)
    
        if len(flag) != len(self.property_names):
            raise ValueError("flag length mismatch with descriptor count")
    
        flag_str = ''.join(['1' if b else '0' for b in flag])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"flag_{flag_str}.csv")
    
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            embeddings = np.array(df['embedding'].apply(ast.literal_eval).to_list())
            print(f"[Cached] Loaded from {save_path}")
            return embeddings
    
        # Compute new embeddings
        vecs: list[np.ndarray] = []
        rows = []
    
        for smi in tqdm(smiles_list, desc=f"Embedding flag {flag_str}"):
            prompt = self._create_prompt(smi, flag)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            vecs.append(vec)
            rows.append({'smiles': smi, 'embedding': vec.tolist()})
    
        # Save to file
        df_save = pd.DataFrame(rows)
        df_save.to_csv(save_path, index=False)
    
        # print(f"[Saved] {save_path}")
        return np.array(vecs)