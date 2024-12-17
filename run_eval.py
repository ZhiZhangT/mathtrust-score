from utils import RUN_Config
from trust_eval import TRUST_SCORE
import nltk

# nltk.download('punkt_tab')

config = RUN_Config()

# Assume you load this from a JSON or YAML file
example_config = {
    # "prompt_file": "prompts/asqa_rejection.json",
    # "eval_file": "amath_qns_eval_data.json",
    "output_dir": "save/AMath",
    "quick_test": None,
    "ndoc": 5,
    "shot": 2,
    "seed": 42,
    "no_doc_in_demo": False,
    "fewer_doc_in_demo": False,
    "ndoc_in_demo": None,
    "no_demo": False,
    "eval_type": "em",
    "model": "openai/gpt-4o",
    "openai_api": False,
    "azure": False,
    "lora_path": None,
    "vllm": False,
    "temperature": 0.1,
    "top_p": 0.95,
    "max_new_tokens": 300,
    "max_length": 128000,
    "num_samples": 1
}

# Update config with new values
config.update_from_dict(example_config)

score = TRUST_SCORE(config)

print(score)