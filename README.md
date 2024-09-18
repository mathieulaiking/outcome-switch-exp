# Primary Outcome Switching experiments

## Corpus 

We extracted the data from the study on selective reporting by [Pellat et al. 2022](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-022-09334-5), we got the full text when possible else the abstract. We filtered the sentences of the text given some regular expression that matches primary outcomes. We annotated it for token classification with primary outcome entities. We then got the associated registry outcomes for each article. And we finally annotated the similarity between the primary outcomes in the article and those declared in the registry.

## Method

We then use a pipeline including a NER transformer model and a Semantic Similarity transformer model (based on [`sentence-transformers`](https://sbert.net/)).
What this pipeline does : 

## Evaluation
