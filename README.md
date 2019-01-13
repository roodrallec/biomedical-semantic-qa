Bio-Medical Question Classification

Author: Jorge Alexander

A multiclass classifier for classifying questions into four types is designed and applied over two parts: The first part consists of applying it to a set of 2,252 bio-medical questions annotated with their type from the BioAsQ challenge task 6b, and the second is to apply it to a dataset of Quora duplicate question pairs, in order to improve the overall classification accuracy of the first part. It is found that randomly initialized embeddings perform better than weighted embeddings on the BioASQ dataset. It is also found that increasing the train data set through model predictions of the Quora duplicate questions improves the performance of the original model, although only for a limited increment in the extended dataset size.

Code Execution:
Create a datasets folder in the root directory and add two files named "BioASQ-trainingDataset6b.json" and "quora_duplicate_questions.tsv":

1. "BioASQ-trainingDataset6b.json" - The training data from Task 6B found on the BioASQ website (you will have to register http://bioasq.org/)


2. "quora_duplicate_questions.tsv" - The tsv file of the publicly available Quora duplicate question dataset from (https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)


