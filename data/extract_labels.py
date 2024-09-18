import os 
import json

for file in os.listdir("data/ap_selective_reporting"):
    fpath = os.path.join("data/ap_selective_reporting", file)
    data = json.load(open(fpath))
    if "filtered_article_outcomes" in data:
        article_outcomes = data["filtered_article_outcomes"]
    elif "article_outcomes_text" in data:
        article_outcomes = data["article_outcomes_text"]
    else:
        article_outcomes = []
    output = {
        "article_sentences": data["article_filtered_sentences"] if "article_filtered_sentences" in data else [],
        "registry_outcomes": data["registry_primary_outcomes"],
        "article_outcomes": article_outcomes,
        "connections":[(c["registry_index"],c["article_index"],c["similar"]) for c in data["article2registry_connections"]] if "article2registry_connections" in data else [],
    }
    json.dump(output, open(fpath.replace("ap_selective_reporting","ap-labels"), "w"), indent=2)