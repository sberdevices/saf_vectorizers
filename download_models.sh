echo "## Installing gdown lib..."
pip install gdown==3.12.2

echo "## Creating static dir for models..."
cd saf_vectorizers
mkdir -p static
cd static

echo "## Downloading SBERT files..."
gdown https://drive.google.com/uc?id=1uay-B3d0VGaoGhVEsVa6AqVn18z9yeyH -O sbert_vocab.txt
gdown https://drive.google.com/uc?id=1LbzFUBBb3hD-dfl6qvYrBokJSmElZxKd -O sbert.graphdef

echo "## Downloading USE embeddings and unpacking it..."
curl -L https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed > universal_sentence_encoder.tar.gz
mkdir -p universal_sentence_encoder
tar -xvf universal_sentence_encoder.tar.gz -C universal_sentence_encoder

echo "## Following files were got:"
echo $(ls)
echo "## Finish!"
