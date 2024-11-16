
### Commands

export GOOGLE_APPLICATION_CREDENTIALS="../wired-glider-441301-e1-b6a080d0f979.json"

python3 preprocess.py

python3 train.py --dataset_path=./processed_data/tokenized_amazon_reviews --output_dir=./

distilbert-sentiment-model

python create_test_dataset.py

python test.py

export TOKENIZERS_PARALLELISM=false
export MLFLOW_TRACKING_URI="http://35.193.202.32:5000"

python test_bias.py

gcloud auth configure-docker us-east1-docker.pkg.dev

docker build -t us-east1-docker.pkg.dev/wired-glider-441301-e1/sentiment-analysis-model/distilbert-sentiment-analysis:1.0.0 .

docker push us-east1-docker.pkg.dev/wired-glider-441301-e1/sentiment-analysis-model/distilbert-sentiment-analysis:1.0.0

