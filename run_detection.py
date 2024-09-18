# author: Mathieu Laï-king
import os
import re
import json
import torch
import pandas
import numpy as np
from nltk import sent_tokenize
from Levenshtein import distance
from typing import List, Dict, Set, Tuple, Union, Callable, Any
from sentence_transformers import util, SentenceTransformer
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


class OutcomeSwitchDetector:

    def __init__(self, 
                 ner_model_path:str,
                 sim_model_path:str,
                 sim_threshold:float):
        model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_path, model_max_length=model.config.max_position_embeddings)
        self.ner_model = pipeline(
            "ner",
            model=model,
            tokenizer=self.tokenizer,
            stride=64, # overlap between chunks, chunk is done only when stride is set 
            aggregation_strategy="average",
        )
        self.sim_model= SentenceTransformer(sim_model_path)
        self.sim_threshold = sim_threshold
    
    def get_tokenized_text(self, text):
        return self.tokenizer.decode(self.tokenizer.encode(text, add_special_tokens=False))

    def preprocess_article(self, full_text:Union[str, Dict[str,List[str]], List[str]]):
        if isinstance(full_text, dict): # sections dict
            preprocessed_text = ""
            for title, content_list in full_text.items():
                preprocessed_text += title + "\n"
                preprocessed_text += " ".join(content_list) + "\n"
            preprocessed_text = sent_tokenize(preprocessed_text)
        elif isinstance(full_text, str): # full text
            preprocessed_text = sent_tokenize(full_text)
        elif isinstance(full_text, list): # list of sentences
            preprocessed_text = full_text
        return preprocessed_text

    def filter_article(self, sentences:List[str]):
        PRIMARY_OUTCOME_REGEX = "(main|primary|principal){1}(\s+\w+){0,3}\s+(end(\s|-)?point|outcome|criterion|measure|parameter|aim|objective|hypothesis){1}"
        return [s for s in sentences if re.search(PRIMARY_OUTCOME_REGEX, s, re.IGNORECASE)]
    
    def extract_article_outcomes(self, sentences:List[str]) -> List[str]:
        outcomes = []
        for sent in sentences :
            ner_results = self.ner_model(sent)
            for entity in ner_results:
                if entity["word"] not in outcomes :
                    outcomes.append(entity["word"])
        return sorted(outcomes)
    
    def filter_article_outcomes(self, outcomes:List[str], threshold:float) :
        embeddings = self.sim_model.encode(outcomes)
        outcomes_copy = outcomes.copy() # to keep the original list intact
        if len(outcomes) >= 2 :
            for i in range(len(outcomes)):
                if outcomes[i] is None :
                    continue
                for j in range(i+1, len(outcomes)):
                    if util.cos_sim(embeddings[i], embeddings[j]) > threshold:
                        if len(outcomes[i]) >= len(outcomes[j]):
                            outcomes_copy[j] = None
                        else :
                            outcomes_copy[i] = None
        filtered_embeddings = np.array([embeddings[i] for i in range(len(outcomes)) if outcomes_copy[i] is not None])
        filtered_text =  [o for o in outcomes_copy if o is not None]
        return filtered_text, filtered_embeddings

    def compare_article_and_registry(self, 
                                     reg_outcomes:List[str],
                                     art_embs:List[torch.Tensor]) -> Set[Tuple[int,int,float]]:
        connections = set()
        reg_embs = self.sim_model.encode(reg_outcomes)
        cosines_scores = util.cos_sim(reg_embs, art_embs)
        for i in range(len(reg_embs)):
            for j in range(len(art_embs)):
                is_similar = cosines_scores[i,j].item() >= self.sim_threshold
                connections.add((i,j,is_similar))
        return connections

    def __call__(self, article_text, registry_outcomes):
        article_outcomes, connections = [], []
        if not article_text:
            return article_outcomes, connections
        # article processings
        preprocessed_article = self.preprocess_article(article_text)
        filtered_sentences = self.filter_article(preprocessed_article)
        article_outcomes = self.extract_article_outcomes(filtered_sentences)
        if not article_outcomes:
            return article_outcomes, connections
        article_outcomes, filtered_article_embeddings = self.filter_article_outcomes(article_outcomes, self.sim_threshold)
        if not article_outcomes:
            return article_outcomes, connections
        # comparison
        connections = self.compare_article_and_registry(registry_outcomes, filtered_article_embeddings)
        return article_outcomes, connections

def is_similar(set1:Set, set2:Set, comparison_function:Callable[[str,str,float],bool]):
    if len(set1) != len(set2) :
        return False
    else : 
        set1_copy = set1.copy()
        set2_copy = set2.copy()
        for elem1 in set1 :
            for elem2 in set2 :
                if comparison_function(elem1, elem2) :
                    if elem1 in set1_copy:
                        set1_copy.remove(elem1)
                    if elem2 in set2_copy:
                        set2_copy.remove(elem2)
        
        return not set1_copy and not set2_copy
    
def compare_levenshtein(string1, string2, threshold=0.4):
    """ This function compares two strings using the Levenshtein distance.
    the threshold is the maximum distance between the two strings to consider them similar.
    """
    d = distance(string1, string2, weights=(1,1,2))
    max_len = max(len(string1), len(string2))
    return d/max_len <= threshold or string1 in string2 or string2 in string1

def convert_to_criterias(connections:Set[Tuple[int,int,float]], pred_outcomes:List[str], registry_outcomes:List[str]):
    criterias = {
        "POS_identif_possible": "",
        "PO_switch": "",
        "PO_not_reported_registry": "",
        "PO_not_reported_publication": "",
        "POS_timeframe_metric": "",
        "Description": ""
    }
    if not pred_outcomes :
        criterias["POS_identif_possible"] = "No"
        return criterias
    else :
        criterias["POS_identif_possible"] = "Yes"


def format_predictions(predictions:Dict[str,Any]):
    output_string = "REGISTRY OUTCOMES :\n * " + "\n * ".join(predictions["registry_outcomes"]) + "\n\n"
    output_string += "TRUE ART OUTCOMES :\n * " + "\n * ".join(predictions["true_article_outcomes"]) + "\n\n"
    output_string += "WRONG ART OUTCOMES :\n"
    for pao in predictions["pred_article_outcomes"]:
        if pao not in predictions["true_article_outcomes"]:
            output_string += f"+ {pao}\n"
    for tao in predictions["true_article_outcomes"]:
        if tao not in predictions["pred_article_outcomes"]:
            output_string += f"- {tao}\n"

    output_string += "\n\nTRUE CONNECTIONS :\n"   
    for ro,tao,is_sim in predictions["true_connections"]:
        if is_sim :
            output_string += f"{predictions['registry_outcomes'][ro]} -> {predictions['true_article_outcomes'][tao]}\n"

    output_string += "\n\nWRONG CONNECTIONS :\n"
    for ro,pao,is_sim in predictions["pred_connections"]:
        if (ro,pao,is_sim) not in predictions["true_connections"]:
            symbol = "+" if is_sim else "-"
            output_string += f"{symbol} {predictions['registry_outcomes'][ro]} -> {predictions['pred_article_outcomes'][pao]}\n"

    return output_string


def detection(
        data_dir,
        ner_path,
        sim_path,
        out_dir
    ):
    # useful variables
    ner_model = ner_path.split("/")[-2]
    sim_model = sim_path.split("/")[-2]
    out_pred_dir = os.path.join(out_dir, f"{ner_model}_{sim_model}")
    if not os.path.exists(out_pred_dir):
        os.makedirs(out_pred_dir)

    # load detector models
    print("Loading models...")
    sim_threshold = pandas.read_csv(sim_path + "/eval/binary_classification_evaluation_results.csv").iloc[-1]["cossim_accuracy_threshold"]
    detector = OutcomeSwitchDetector(ner_path, sim_path, sim_threshold)

    # Processings
    perfect_match = 0
    approximate_match = 0
    file_nb = len(os.listdir(data_dir))
    for i, fname in enumerate(os.listdir(data_dir)):
        print("Processing file {}/{}...".format(i+1, file_nb), end="\r")
        # read input
        fpath = os.path.join(data_dir, fname)
        input_dict = json.load(open(fpath, "r"))
        article_sentences = input_dict["article_sentences"]
        registry_outcomes = input_dict["registry_outcomes"]
        # read true output
        true_article_outcomes = sorted(input_dict["article_outcomes"])
        true_connections = input_dict["connections"]
        # predict output
        pred_outcomes, pred_connections = detector(article_sentences, registry_outcomes)
        # evaluate
        true_article_outcomes = [detector.get_tokenized_text(a) for a in input_dict["article_outcomes"]]
        true_connections  = set([(c[0],c[1],c[2]) for c in input_dict["connections"]])
        perf_bool =  set(pred_outcomes) == set(true_article_outcomes) and set(pred_connections) == set(true_connections)
        approx_bool =  is_similar(set(pred_outcomes),true_article_outcomes, compare_levenshtein) and set(pred_connections) == set(true_connections)
        perfect_match += 1 if perf_bool else 0
        approximate_match += 1 if approx_bool else 0
        # save errors
        if not approx_bool:
            wrong_pred = {
                "registry_outcomes" : registry_outcomes,
                "true_article_outcomes" : list(true_article_outcomes),
                "true_connections" : list(true_connections),
                "pred_article_outcomes" : pred_outcomes,
                "pred_connections" : list(pred_connections),
                "approximate_match" : approx_bool
            }
            output_path = os.path.join(out_pred_dir, fname)
            with open(output_path, "w") as f:
                json.dump(wrong_pred, f, indent=4, default=str)
            wrong_pred_string = format_predictions(wrong_pred)
            with open(output_path.replace(".json",".txt"), "w") as f:
                f.write(wrong_pred_string)
    print(f"Processed {file_nb} files.")
    performances = {
        "ner_model" : ner_model,
        "sim_model" : sim_model,
        "sim_threshold" : sim_threshold,
        "perfect_match" : perfect_match,
        "approximate_match" : approximate_match
    }
    with open(os.path.join(out_dir, f"{ner_model}_{sim_model}_performances.json"), "w") as f:
        json.dump(performances, f, indent=4, default=str)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="path to the data directory")
    parser.add_argument("--ner_path", type=str, required=True, help="path to the NER model")
    parser.add_argument("--sim_path", type=str, required=True, help="path to the similarity model")
    parser.add_argument("--out_dir", type=str, required=True, help="path to the output directory")
    args = parser.parse_args()
    detection(**vars(args))

if __name__ == "__main__":
    main()