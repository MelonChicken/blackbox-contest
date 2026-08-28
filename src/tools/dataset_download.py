import kagglehub
# https://www.kaggle.com/datasets/johnmageetud/recaptured-identity-documents
# Download latest version
# path = kagglehub.dataset_download("johnmageetud/recaptured-identity-documents")
#
# print("Path to dataset files:", path)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("handsomeWilliam/Relation252K")