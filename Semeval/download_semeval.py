from datasets import load_dataset
from huggingface_hub import login
import json
import csv

hf_token = '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'
login(hf_token)
dataset = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
with open("id2rel.json","rb") as f:
    id2rel = json.load(f)

fields_csv = ["paragraph_id","paragraph","predicate_name","subject_names","object_names"]

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
        rel[relid].append([str(i),context,relation_label,subj,obj])
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

save_preprocessed_docs(dataset['train'],False)
print('Processed train dataset')
save_preprocessed_docs(dataset['test'],True)
print('Processed test dataset')
