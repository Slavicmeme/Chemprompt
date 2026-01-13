# ChemPrompt: LLM-based Molecular Representation via Chemical Descriptors

ChemPrompt is a framework that integrates Large Language Models (LLMs) with RDKit molecular descriptors to enhance molecular representation learning for QSAR analysis and descriptor selection studies.

---

## Environment Setup
To set up the environment for ChemPrompt, clone the repository and create the conda environment:

```bash
git clone {url}
cd Chemprompt
conda env create -f environment.yml
```

Then, set the LLM cache path to specify where models will be stored:

```bash
echo 'export TRANSFORMERS_CACHE={directory_path}' >> ~/.bashrc
source ~/.bashrc
```

---

## 5.1 Prompt with Descriptors is better than Ones with No Descriptor
This section compares full descriptor-informed prompts against prompts without any descriptor information.  
The experiments are conducted using two notebooks: **full_descriptors.ipynb** and **no_description.ipynb**.  

The variable **DATASETS** specifies the list of datasets to iterate through, and **MODEL_LIST** defines the model repositories and names in the following format:  
`{"repo": "model_repo", "name": "model_name"}`.  
The performance difference between descriptor-informed and descriptor-free prompts is then evaluated across datasets and models.

---

## 5.2 Existence of Optimal Subset of Descriptors
This section tests whether there exists a subset of descriptors that yields strong performance.  
Random subsets of 10, 30, 50, 70, and 90 descriptors are sampled 100 times each.  

The configuration variable **RANDOM_PLAN** defines how many random samples to evaluate for each descriptor count:
```python
RANDOM_PLAN = {
    10: 100,
    30: 100,
    50: 100,
    70: 100,
    90: 100
}
```
Each mapping condition is iterated, and performance is evaluated to investigate whether smaller subsets of descriptors can match or surpass the full descriptor set.

---

## 5.3 Discovery of Optimal Subset of Descriptors via Genetic Algorithm (GA)
This section introduces a feature selection process using statistical ranking followed by a Genetic Algorithm (GA).  
After descriptor filtering, univariate analysis (**f_regression**) is performed between each descriptor value and label.  
Descriptors are ranked by statistical significance, and the top **K = 50** features are selected to initialize the GA process.  

For each data fold, GA optimization is executed to identify an optimal subset of descriptors.  
The final optimal subsets and their corresponding model performances are evaluated and compared across datasets.

After the GA optimization completes, the results for each fold are automatically saved under the following directory:  
`evolutionary_process/{dataset}_fold_{fold_num}_GA_{mode}/best_overall_metrics.csv`

This CSV file contains the **flag**, **full_flag**, and corresponding **evaluation metrics** for the optimal subset discovered by the GA process.  

- **flag** represents a binary mask indicating the combination of selected descriptors within the feature selection space (e.g., top-K filtered descriptors).  
- **full_flag** represents the same binary mask projected back onto the full descriptors space, reflecting which descriptors are active in the original full-dimensional context.  
- The reported **metrics** (e.g., RMSE, R², Pearson, Spearman) correspond to the model performance obtained after generating embeddings based on the selected descriptors and training the regression model accordingly.  

The **full_flag** will be utilized in Section **5.6 Selection**

---

## 5.4 ChemPrompt Outperformed Traditional Methods
This section compares ChemPrompt with conventional machine learning and neural network approaches.  

Three types of experiments are included:
- **Fingerprint.ipynb**: Extracts molecular fingerprints and uses them as features for machine learning models.  
- **RNN.ipynb**: Uses SMILES features with RNN and LSTM architectures for sequence-based learning.  
- **Transformer.ipynb**: Loads pretrained chemical Transformer models to extract SMILES embeddings for machine learning model input.  

Results demonstrate that ChemPrompt outperforms these traditional approaches by effectively combining molecular descriptors with LLM embeddings.

---

## 5.5 LLM Inference is Inaccurate in QSAR
This section examines the accuracy of LLM inference in QSAR-style question answering.  
Each notebook (**{MODEL_NAME}.ipynb**) loads a specific Hugging Face model and performs Q&A-style prediction following the model’s recommended prompt template.  
Prompt instruction formats differ across datasets.  

Before running **GPT-OSS.ipynb**, please update the Transformers library to the required version:
```bash
pip install --upgrade transformers==4.56.2
```
This ensures compatibility with the GPT-OSS model architecture.

---

## 5.6 No Degradation of ChemPrompt with Quantization
This section evaluates the robustness of ChemPrompt under numerical quantization using the optimal descriptor subsets discovered in Section 5.3.  
**Quantization.ipynb** loads the model in different precisions and compares performance:  
“full” corresponds to **float32**, and “half” corresponds to **float16**.  

---

## System Requirements and Contact
All experiments were conducted on **Ubuntu 22.04 LTS**.  
It is recommended to use at least **20 GB VRAM** for float16 precision and **40 GB VRAM** for float32 precision when working with 8B-scale models.  

For questions regarding embedding extraction from GPT-OSS models, please contact:  
**Email:** bbq9088@gmail.com
