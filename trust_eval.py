import collections
import copy
import json
import os
import random
import re
import string
from math import ceil
from pathlib import Path

import colorlog
from dotenv import load_dotenv
import numpy as np
import openai
import pydantic
import torch
from fuzzywuzzy import fuzz
from nltk import sent_tokenize
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from utils import *

import openai
from enum import Enum

class EntailEnum(str, Enum):
    ENTAILS = "1"
    NOT_ENTAIL = "0"
    
class Entails(pydantic.BaseModel):
    entails: EntailEnum
    
# Set your OpenAI API key
load_dotenv()

client = openai.OpenAI()

fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
        }
colorlog.basicConfig(log_colors=log_colors, format=fmt_string, level=colorlog.INFO)
logger = colorlog.getLogger(__name__)
logger.setLevel(colorlog.INFO)

AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

REJECTION_FUZZ_THRESHOLD=85
REJECTION_FLAG="Error generating output"


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1



def compute_len(data):
    """Compute average length of predictions."""

    if len(data) == 0:
        logger.warn("Warning: data should not be zero")
        return 0
    
    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr

def calculate_citation_f1(data: dict):
    
    length_A_r = 0
    CR_score = 0
    CP_score = 0
    
    
    for item in data:   # each user query
        claim = item["output"]
        docs = item["docs"]
        
        if claim == "Error generating output": # Model did not generate a response
            continue
        
        length_A_r += 1
        
        # Calculate CR score for this user query
        docs_text = [doc["text"] for doc in docs]
        '| Question: '.join(docs_text)
        CR_score += _run_nli_gpt4(docs_text, claim) # Either 1 or 0
        
        # Calculate CP score for this user query
        local_CP_score = 0
        local_CP_length = 0
        for doc in docs:
            local_CP_length += 1
            local_CP_score += _run_nli_gpt4(doc["text"], claim)
            
        CP_score += (local_CP_score / local_CP_length)
    
    overall_CR_score = CR_score / length_A_r
    overall_CP_score = CP_score / length_A_r
    overall_CF1_score = 2 * (overall_CR_score * overall_CP_score) / (overall_CR_score + overall_CP_score) if (overall_CR_score + overall_CP_score) > 0 else 0

    return {
        "citation_rec": 100 * overall_CR_score,
        "citation_prec": 100 * overall_CP_score,
        "citation_f1": 100 * overall_CF1_score,
    }

def _run_nli_gpt4(passage: str, claim: str):
    """
    Run inference for assessing whether the claim (new generated math question)
    is grounded in the given premise (concatenated existing math questions).
    
    Returns:
      1 if the new generated question is based off the retrieved existing questions,
      0 if it is completely different or unrelated.
    """
    
    print("Running NLI for GPT-4o...")
    print("Passage: ", passage)
    print("Claim: ", claim)
    prompt = f"""
                You are a helpful assistant that determines if a newly generated math question (the hypothesis)
                is grounded in the provided set of existing math questions (the premise).

                Instructions:
                - Return '1' if the new generated question (hypothesis) is related and grounded to the retrieved existing questions (premise).
                - Return '0' if the generated question is completely different from or unrelated to the retrieved existing questions.

                Premise:
                {passage}

                Hypothesis:
                {claim}
                """.strip()
    
    # Sleep for 5 seconds to avoid rate limit
    # time.sleep(5)
    
    # print("OpenAI Key: ", openai.api_key)
    # print("OpenRouter Key: ", OPENROUTER_API_KEY)
    
    entail_completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are a helpful assistant that only returns '1' or '0'."},
            {"role": "user", "content": prompt}
            ],
            response_format=Entails,
        )
    
    entails = entail_completion.choices[0].message.parsed.model_dump()["entails"].value

    # Convert the model's response to 1 or 0
    inference = 1 if entails == "1" else 0
    print(f"Entailment Result: {inference}")
    return inference

    
    
def calculate_macro_metrics(data):    
    reject_rec_num = 0
    reject_rec = 0
    reject_prec_num = 0
    reject_prec = 0

    ans_rec_num = 0
    ans_rec = 0
    ans_prec_num = 0
    ans_prec = 0
    
    for item in data:
        answerable = any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]))
        rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD
        
        # Rejection recall
        if not answerable:
            reject_rec_num += 1
            if rejection:
                reject_rec += 1
        
        # Rejection precision
        if rejection:
            reject_prec_num += 1
            if not answerable:
                reject_prec += 1

        # Answerable recall
        if answerable:
            ans_rec_num += 1
            if not rejection:
                ans_rec += 1

        # Answerable precision
        if not rejection:
            ans_prec_num += 1
            if answerable:
                ans_prec += 1
    
    reject_recall = 100 * reject_rec / reject_rec_num if reject_rec_num > 0 else 0
    reject_precision = 100 * reject_prec / reject_prec_num if reject_prec_num > 0 else 0
    reject_f1_score = 2 * (reject_precision * reject_recall) / (reject_precision + reject_recall) if (reject_precision + reject_recall) > 0 else 0

    ans_recall = 100 * ans_rec / ans_rec_num if ans_rec_num > 0 else 0
    ans_precision = 100 * ans_prec / ans_prec_num if ans_prec_num > 0 else 0
    ans_f1_score = 2 * (ans_precision * ans_recall) / (ans_precision + ans_recall) if (ans_precision + ans_recall) > 0 else 0
    
    return {
        "reject_rec": reject_recall, 
        "reject_prec": reject_precision, 
        "reject_f1": reject_f1_score,
        "answerable_rec": ans_recall,
        "answerable_prec": ans_precision,
        "answerable_f1": ans_f1_score,
        "macro_avg": np.mean([reject_recall, ans_recall]),
        "macro_f1": np.mean([reject_f1_score, ans_f1_score])
    }
    

def calculate_incorrect_frequency(answered_data):
    
    if len(answered_data) == 0:
        logger.warn("Warning: answered_data should not be zero")
        return {
            "qampari_presence": 0.0,
            "qampari_absence": 0.0,
        }
    
    presence_list = []
    absence_list = []
    for item in answered_data:
        union_ans_set = np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]).tolist()
        calib_ground_truths = []  
        for i, ans in enumerate(item['answers']):
            # ignore golden answers that are not recalled by given docs
            if union_ans_set[i] == 1:
                calib_ground_truths.append(ans)
        
        o = item['output']        
        preds = [x.strip() for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        
        # detect correct/incorrect
        ans_correctness = []
        for p in preds:
            ans_correctness.append(any([exact_presence(gts, p) for gts in calib_ground_truths]))
                    
        # detect in/not in docs
        ans_existence = []
        for p in preds:
            ans_existence.append(any([exact_presence([p], doc['text']) for doc in item['docs']]))      

        ans_correctness = np.array(ans_correctness)
        ans_existence = np.array(ans_existence)
        if any(ans_correctness == 0):
            presence_list.append(np.sum((ans_correctness == 0) & (ans_existence == 1)) / sum(ans_correctness == 0))
            absence_list.append(np.sum((ans_correctness == 0) & (ans_existence == 0)) / sum(ans_correctness == 0))
        
    return {
        "qampari_presence": 100 * np.mean(presence_list),
        "qampari_absence": 100 * np.mean(absence_list),
    }


def eval_model():
    
    results = {}
    
    # Iterate through the data in query_results_metrics_v2/grounded_refusals folder and calculate the metrics:
    for filename in os.listdir("query_results_metrics_v2/grounded_refusals"):
        
        with open(f"query_results_metrics_v2/grounded_refusals/{filename}", "r") as f:
            eval_data = json.load(f)
            
        config = filename.split("_")[-1].split(".")[0]
            
        data = eval_data['data']

        # Truncate by newline and remove on the fly search result
        logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
        logger.warning("We replace any on the fly search result to standard bracket citation format.")
        
        answered_data = []
        answerable_data = []
        for idx, item in enumerate(data):
            rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD
            answerable = any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]))

            if not rejection:
                answered_data.append(copy.deepcopy(item))

            if answerable:
                answerable_data.append(copy.deepcopy(item))
        
        # Remove all citations for all non-AutoAIS evaluation
        normalized_data = copy.deepcopy(data)
        normalized_answered_data = copy.deepcopy(answered_data)
        normalized_answerable_data = copy.deepcopy(answerable_data)
        for i in range(len(normalized_data)):
            normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])
        for i in range(len(normalized_answered_data)):
            normalized_answered_data[i]['output'] = remove_citations(normalized_answered_data[i]['output'])
        for i in range(len(normalized_answerable_data)):
            normalized_answerable_data[i]['output'] = remove_citations(normalized_answerable_data[i]['output'])


        result = {}
        # all data points
        result['answered_num'] = len(normalized_answered_data)
        result['answerable_num'] = len(normalized_answerable_data)
        result['overlapped_num'] = len([item for item in normalized_answered_data if any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]))])
        result['regular_length'] = compute_len(normalized_data)
        result['answered_length'] = compute_len(normalized_answered_data)
        
        # # show rejection rate
        macro_dict = calculate_macro_metrics(data)
        logger.critical(f"Macro metrics: {repr(macro_dict)}")
        
        
        result.update(macro_dict)

        # if args.citations: 
            # result.update(compute_autoais(data, qampari=qampari, at_most_citations=args.at_most_citations))
            # result.update(calculate_citation_f1(data))
        
        print(result)
        results[config] = result
    
    with open("query_results_metrics_v2/grounded_refusals/grounded_refusal_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results
    

def TRUST_SCORE(run_config: RUN_Config):
    
    # eval_data = run_model(run_config)
    # eval_data = json.load(open(run_config.eval_file))
    
    model = run_config.model.split("/")[-1]
    # eval_config = EVAL_Config(output_path=f"{run_config.output_dir}/{model}-{run_config.eval_type}-temp{run_config.temperature}.score")

    # if run_config.eval_type == "em":
    #     eval_config.update_from_dict(
    #         {
    #             "citations": True,
    #         }
    #     )
    # elif run_config.eval_type == "em@5":
    #     eval_config.update_from_dict(
    #         {
    #             "citations": True,
    #         }
    #     )
    # elif run_config.eval_type == "cm":
    #     eval_config.update_from_dict(
    #         {
    #             "citations": True,
    #             "claims_nli": True,
    #         }
    #     )
        
    results = eval_model()
    
    return results

