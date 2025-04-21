# Example using Python
from huggingface_hub import HfApi, upload_file

# Assumes you are logged in via huggingface-cli login

# Option 1: Upload a single file
upload_file(
    path_or_fileobj="./model/model.keras",
    path_in_repo="model.keras", # Path within the HF repo
    repo_id="chaoder/tumor-classifier", # Your repo name
    repo_type="model" 
)
print("Model uploaded to Hugging Face Hub!")

# Option 2: Upload the whole model directory (if you have other related files)
# from huggingface_hub import upload_folder
# upload_folder(
#     folder_path="./model",
#     path_in_repo="model", # Uploads the 'model' folder itself
#     repo_id="your-username/tumor-classifier",
#     repo_type="model"
# )