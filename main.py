from huggingface_hub.utils import HFValidationError
import os
from transformers import AutoTokenizer, AutoModel
import os
os.environ['HF_HOME'] = 'pretrain'
model_name = "google-bert/bert-base-multilingual-cased"

pretrain_path = "pretrain_model"
model_name = "google-bert/bert-base-multilingual-cased"

model_path = os.path.join(pretrain_path, model_name)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, resume_download=None)
    model = AutoModel.from_pretrained(model_path, resume_download=None)
except HFValidationError as e:
    print(f"模型加载失败")
except Exception as error:
    print(error)