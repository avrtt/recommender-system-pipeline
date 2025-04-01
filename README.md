This repository demonstrates an implementation of a recommender system pipeline using PyTorch. The system covers the full pipeline from offline data processing and model training to online serving and prediction.

The pipeline was made as part of my freelance project for demonstation purposes. 

## Dependencies & tools
- **Conda** for offline data processing and model training with pandas and PyTorch
- **Docker** to run Redis, Elasticsearch, Feast, and Triton without affecting the local environment
- **Flask** simulates the backend, acting as a RESTful web server
- **Redis** stores user terms and vectors for recall
- **Elasticsearch** manages the term and vector indices for item recall
- **Feast** handles feature storage for online ranking
- **Triton** serves real-time model predictions

## Architecture
This RS follows a three-phase development process:
1. **Offline**: Preprocessing, model training, and feature engineering
2. **Offline-to-Online**: Transition from offline data to online services using Docker containers
3. **Online**: Real-time serving and recommendation through a Flask web service

<br>

## Offline part

### Conda environment setup
Create a Conda environment for offline processing and model training:
```bash
conda create -n rsppl python=3.8
conda activate rsppl
conda install --file requirements.txt --channel anaconda --channel conda-forge
```

### Preprocessing
Split the dataset, process labels, and transform features:
```bash
cd offline/preprocess/
python s1_data_split.py
python s2_term_trans.py
```
- **Labels**: Convert the MovieLens dataset to implicit feedback (positive if rating > 3)
- **Samples**: Reserve the last 10 rated items per user for online evaluation
- **Features**: Encode sparse features as integer sequences

### Term and vector recall
Generate term and vector recalls based on user interactions and factorization machines:
```bash
cd offline/recall/
python s1_term_recall.py
python s2_vector_recall.py
```
- **Term Recall**: Matches user preferences based on item attributes (e.g., genres)
- **Vector Recall**: Trains a Factorization Machine model for user-item interactions

### Feature engineering and ranking with DeepFM
Train the ranking model with a mix of sparse and dense features:
```bash
cd offline/rank/
python s1_feature_engi.py
python s2_model_train.py
```
- **DeepFM Model**: Trains using one-hot, multi-hot, and dense features. Modified for more flexible embedding dimensions and feature handling

<br>

## Offline-to-online part

### Docker setup
Ensure Docker is installed and pull necessary images:
```bash
docker pull redis:6.0.0
docker pull elasticsearch:8.8.0
docker pull feastdev/feature-server:0.31.0
docker pull nvcr.io/nvidia/tritonserver:20.12-py3
```

### Load user data into Redis
Load user terms and vectors into Redis for fast retrieval:
```bash
docker run --name redis -p 6379:6379 -d redis:6.0.0
cd offline_to_online/recall/
python s1_user_to_redis.py
```

### Load item data into Elasticsearch
Set up Elasticsearch to handle both term and vector-based recall:
```bash
docker run --name es8 -p 9200:9200 -it elasticsearch:8.8.0
docker start es8
cd offline_to_online/recall/
python s2_item_to_es.py
```

### Load features into Feast
Feast acts as the feature store for the ranking phase:
```bash
cd offline_to_online/rank/
python s1_feature_to_feast.py
docker run --rm --name feast-server -v $(pwd):/home/hs -p 6566:6566 -it feastdev/feature-server:0.31.0
```
Initialize Feast and load features:
```bash
feast apply
feast materialize-incremental "$(date +'%Y-%m-%d %H:%M:%S')"
feast serve -h 0.0.0.0 -p 6566
```

### Convert PyTorch model to ONNX and deploy to Triton
Convert the PyTorch ranking model to ONNX format for serving with Triton:
```bash
cd offline_to_online/rank/
python s2_model_to_triton.py
docker run --rm -p8000:8000 -p8001:8001 -v $(pwd)/:/models/ nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models/
```
Validate consistency between offline and online scores:
```bash
python s3_check_offline_and_online.py
```

<br>

## Online part

### Flask web server
Start the Flask web server for handling recommendation requests:
```bash
conda activate rsppl
cd online/main
flask --app s1_server.py run --host=0.0.0.0
```

### Test the recommender system
Simulate a client request to the Flask server, receiving the top 50 recommended items:
```bash
cd online/main
python s2_client.py
```
