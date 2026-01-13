# config/model_mapping.py

# Model repository mapping
# MODEL_REPO_MAP = {
#    "Llama2": {"repo": "meta-llama", "name": "Llama-2-7b-hf"},
#    "Llama3": {"repo": "meta-llama", "name": "Meta-Llama-3.1-8B"},
#    "Gemma": {"repo": "google", "name": "gemma-2b"},
#    "Mistral": {"repo": "mistralai", "name": "Mistral-7B-v0.1"},
#    "Falcon": {"repo": "tiiuae", "name": "falcon-7b"}
#}

def get_model_repo(model_name, custom_repo=None):
    """
    Returns the Hugging Face repository information for the given model name.
    
    Parameters:
        model_name (str): Name of the model (e.g., "Llama2").
        custom_repo (str, optional): Custom Hugging Face repository if not pre-mapped.

    Returns:
        dict: Dictionary containing 'repo' and 'name' of the model.
    """
    if model_name in MODEL_REPO_MAP:
        return MODEL_REPO_MAP[model_name]
    elif custom_repo:
        repo, name = custom_repo.split('/')
        return {"repo": repo, "name": name}
    else:
        raise ValueError(
            f"Model '{model_name}' not found in the predefined mapping. "
            f"Provide a valid model name or custom repository."
        )
