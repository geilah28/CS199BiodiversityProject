# CS 199 Special Project

A semantic search engine for the Philippine Biodiversity using Named Entity Recognition and Relation Extraction

## Installation
To install and run the application, you will need to have the following dependencies installed on your computer:

-   Node.js and npm (the Node.js package manager)
-   Python 3.9
-   Elasticsearch

### Node.js and npm
Make sure you have Node.js and npm installed. You can check if you have them installed by running the following commands in your terminal:
```
node -v
npm -v
```
If you don't have Node.js and npm installed, you can download and install them from the official website ([https://nodejs.org/](https://nodejs.org/)) or using a package manager like Homebrew (for macOS) or Chocolatey (for Windows).

### Python 3.9
Make sure you have Python 3.x installed. You can check if you have it installed by running the following command in your terminal:

```python --version```

### Elasticsearch

Elasticsearch is a search engine based on the Lucene library. It provides a distributed, multitenant-capable full-text search engine with an HTTP web interface and schema-free JSON documents.

To install Elasticsearch, follow the instructions on the official website ([https://www.elastic.co/downloads/elasticsearch](https://www.elastic.co/downloads/elasticsearch)). You can also install Elasticsearch using a package manager like Homebrew (for macOS) or Chocolatey (for Windows).

Once Elasticsearch is installed, you can start the Elasticsearch server by running the following command in your terminal:

    docker pull elasticsearch:7.9.1

    docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.9.1

Verify that the Elasticsearch container is running by making a GET request to the  `http://localhost:9200`  endpoint:

    curl http://localhost:9200

You should see a response similar to the following:
```
{
  "name" : "57f522a75f62",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "xvS5byPxQTeJZ5qgKasZ7w",
  "version" : {
    "number" : "7.9.1",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "083627f112ba94dffc1232e8b42b73492789ef91",
    "build_date" : "2020-09-01T21:22:21.964974Z",
    "build_snapshot" : false,
    "lucene_version" : "8.6.2",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}
```

### Clone the repository

To get started, clone the repository to your local machine:

Copy code

```
git clone https://github.com/CS-199-Biodiversity-Project/Backend.git
```

This will create a local copy of the repository on your computer.

### Install the dependencies (Backend)

Navigate to the project directory:

```
cd Backend
```

Create a virtual environment
```
python3 -m venv venv 
or 
python -m venv venv
```

Activate the virtual environment
```
venv\Scripts\Activate
```

While inside the virtual environment
```
pip install -r requirements.txt
```

### Install the dependencies (Frontend)
Use another terminal and navigate to the project directory:

```
cd Backend
```

Navigate to the frontend directory:
```
cd frontend
```

Inside the frontend directory
```
npm install
```

### Adding the NER Model and QA Model
Go to Google Drive Link [Google Drive](https://drive.google.com/drive/folders/16g09ztf4K6AjZUADcxI0hTFzXS8uo2J8?usp=sharing)

Download qa_model and ner_model (this will be a zipped file)

(For both models)
After downloading, extract the file 

Copy the folder and place the folder inside the project directory

You should see the project directory folder as:
```
    .
    ├── copious-dataset                 
    ├── frontend                    
    ├── venv    
    ├── qa_model
    │   ├── config.json
    │   ├── pytorch_model.bin
    ├── ner_model
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   ├── special_tokens_map.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── training_args.bin
    │   ├── vocab.txt
    ├── .flaskenv
    ├── .gitmodules
    ├── backend.py
    ├── requirements.txt
    └── README.md
```

### Start the application
First, make sure that docker is already running
   
#### Backend
From the virtual environment,
```
flask run
```
   
#### Frontend
From the frontend directory,
```
npm start
```
   
## Usage
To use the application, simply enter a query like "pelagic sea snakes in Australian Coastal Waters" and click the search button. 
The application will display the entities and their type (from the query) and a question and answer if those two entities have a relationship. 
Only use two entities as the application cannot detect more than two.
