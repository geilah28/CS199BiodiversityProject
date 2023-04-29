import elasticsearch
from elasticsearch import Elasticsearch, helpers
import os
import glob
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}}, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

#Set up ElasticSearch
def insert_passages(): #Insert passages
  folder_path_test = 'copious-dataset/test'
  folder_path_dev = 'copious-dataset/dev'
  folder_path_train = 'copious-dataset/train'

  passages = []
  for filename in sorted(glob.glob(os.path.join(folder_path_test, '*.txt'))):
    with open(filename, 'r', encoding="utf-8") as f:
      data = f.read().replace('\n', ' ')
      passages.append(data.lower())

  for filename in sorted(glob.glob(os.path.join(folder_path_dev, '*.txt'))):
    with open(filename, 'r', encoding="utf-8") as f:
      data = f.read().replace('\n', ' ')
      passages.append(data.lower())

  for filename in sorted(glob.glob(os.path.join(folder_path_train, '*.txt'))):
    with open(filename, 'r', encoding="utf-8") as f:
      data = f.read().replace('\n', ' ')
      passages.append(data.lower())

  data= {'passage': passages}
  df=pd.DataFrame(data=data)
  
  return df

dataframe= insert_passages()
es = elasticsearch.Elasticsearch(hosts=["http://localhost:9200"])

Settings = {
    "settings":{
        "number_of_shards":1,
        "number_of_replicas":0
    },
    "mappings":{
        "properties":{
            "passage":{
                "type":"text"
            }
        }
    }
}

def json_formatter(dataset, index_name, index_type='_doc'):
    try:
        List = []
        columns = dataset.columns
        for idx, row in dataset.iterrows():
            dic = {}
            dic['_index'] = index_name
            dic['_type'] = index_type
            source = {}
            for i in dataset.columns:
                source[i] = row[i]
            dic['_source'] = source
            List.append(dic)
        return List
    
    except Exception as e:
        print("There is a problem: {}".format(e))

MY_INDEX = es.indices.create(index="content", ignore=[400,404], body=Settings)
json_Formatted_dataset = json_formatter(dataset=dataframe, index_name='content', index_type='_doc')

try:
    res = helpers.bulk(es, json_Formatted_dataset)
    print("successfully imported to elasticsearch.")
except Exception as e:
    print(f"error: {e}")


#Setup NER
NER_checkpoint = "./ner_model"
token_classifier = pipeline("token-classification", model=NER_checkpoint, aggregation_strategy="first")

def named_entity_recognition(query):
    if (query)!="":
        ner_output= token_classifier(query)
       
        if len(ner_output)<2:
            return "There must be two entities in the input"
        else:
            entity_group=[]
            entities=[]
        
            for x in range(len(ner_output)):
                entity_group.append(ner_output[x]["entity_group"])
                entities.append(ner_output[x]["word"])

            return jsonify(entity1=entities[0] , entity2=entities[1] , type1=entity_group[0], type2=entity_group[1])

    else:
       return "Please provide input"

@app.route("/ner", methods=["GET"])
def ner_route():
  # get the search query from the request query parameters
    input = request.args.get("input")
    entities= named_entity_recognition(input)

    return entities

#Set up QA or RE
def question_answer(entity_group, entities):
    questions=[]
   
    for x in range(len(entity_group)):
        for y in range(len(entity_group)):
            if entity_group[x]=="Taxon" and entity_group[y]=="Habitat":
                questions.append("Is there %s at %s?" % (entities[x], entities[y]))
            if entity_group[x]=="Taxon" and entity_group[y]=="GeographicalLocation":
                questions.append("Is there %s at %s?" % (entities[x], entities[y]))
            if entity_group[x]=="Habitat" and entity_group[y]=="GeographicalLocation":
                questions.append("Is there %s at %s?" % (entities[x], entities[y]))
            if entity_group[x]=="Taxon" and entity_group[y]=="TemporalExpression":
                questions.append("Did %s happen on %s?" % (entities[x], entities[y]))
            if entity_group[x]=="Habitat" and entity_group[y]=="TemporalExpression":
                questions.append("Did %s happen on %s?" % (entities[x], entities[y]))

    questions = list(dict.fromkeys(questions))
    return questions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RE_checkpoint = "./qa_model"
model = AutoModelForSequenceClassification.from_pretrained(RE_checkpoint)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("roberta-large") 

def predict(question, passage):
    sequence = tokenizer.encode_plus(question, passage, return_tensors="pt", truncation='longest_first')['input_ids'].to(device)
    
    logits = model(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    proba_yes = round(probabilities[1], 2)
    proba_no = round(probabilities[0], 2)

    if proba_yes>proba_no:
       return "Yes"
    elif proba_yes<proba_no:
       return "No"
    else:
       return "Neither"

@app.route("/re", methods=["GET"])
def re_route():
    entitygroup=[]
    entities=[]

    entity1 = request.args.get("entity1")
    entity2 = request.args.get("entity2")
    type1 = request.args.get("type1")
    type2 = request.args.get("type2")
    doc = request.args.get("doc")

    entitygroup.extend([type1,type2])
    entities.extend([entity1,entity2])
    questions= question_answer(entitygroup, entities)


    if len(questions)<1:
        return "The input pair does not show a relation between each other"
    else:
        return  predict(questions[0], doc)

if __name__ == "__main__":
  app.run(debug=True)