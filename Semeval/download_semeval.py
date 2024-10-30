from datasets import load_dataset
from huggingface_hub import login
import json
import csv
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import pandas as pd

hf_token = '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'
login(hf_token)
dataset = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
with open("id2rel.json","rb") as f:
    id2rel = json.load(f)

fields_csv = ["paragraph_id","paragraph","predicate_name","subject_names","object_names"]

def format(string):
    return f"['{string}']"

def save_preprocessed_docs(data, dev : bool):
    rel = {}
    for i,sample in enumerate(data):
        relid = sample['relation']
        relation_label = id2rel[str(relid)]
        obj = sample['sentence'].split("<e1>")[1].split("</e1>")[0]
        subj = sample['sentence'].split("<e2>")[1].split("</e2>")[0]
        context = sample['sentence'].replace("<e1>",'').replace("</e1>",'').replace("<e2>",'').replace("</e2>",'')
        if relid not in rel:
            rel[relid] = []
        rel[relid].append([str(i),context,relation_label,format(subj),format(obj)])
    for key in rel:
        # semeval has a test dataset, not dev, but we keep the name so folder names are the same for all datasets
        if not dev:
            path = f"relation_docs/{key}.csv"
        else:
            path = f"relation_docs_dev/{key}.csv"
        with open(path,"w+") as f:
            writer = csv.writer(f)
            writer.writerow(fields_csv)
            writer.writerows(rel[key])

def get_embeddings(data, dev:bool):
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length=512
    print("model is loaded")

    list_pars = [doc["sentence"] for doc in data]
    list_par_ids = [i for i in range(len(data))]

    embeddings = model.encode(list_pars, show_progress_bar=True, normalize_embeddings=True, batch_size=128)

	#We also keep the length of each doc (i.e. num_words) in case we need a filtering later on 
    #For this dataset, there is only one sentence per document
    num_words = np.array([len(x['sentence']) for x in data])
    if dev:
        path = "embeddings_dev.json"
    else:
        path = "embeddings_train.json"
    with open(path, 'wb') as fOut:
	    pickle.dump({'paragraph_ids':list_par_ids, 'embeddings':embeddings, 'num_words':num_words}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

save_preprocessed_docs(dataset['train'],False)
print('Processed train dataset')
save_preprocessed_docs(dataset['test'],True)
print('Processed test dataset')
get_embeddings(dataset['train'],False)
print('Got train dataset embeddings')
get_embeddings(dataset['test'],True)
print('Got test dataset embeddings')
