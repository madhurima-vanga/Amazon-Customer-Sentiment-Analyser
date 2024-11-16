
### Commands

export GOOGLE_APPLICATION_CREDENTIALS="../wired-glider-441301-e1-b6a080d0f979.json"

python preprocess.py

python train.py --dataset_path=./processed_data/tokenized_amazon_reviews --output_dir=./albert-sentiment-model

python create_test_dataset.py

python test.py

python create_test_bias_dataset.py

python test_bias.py