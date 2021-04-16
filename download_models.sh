echo "Installing gdown lib..."
pip install gdown==3.12.2
cd saf_vectorizers
mkdir -p static
cd static
echo "Downloading SBERT files..."
gdown https://drive.google.com/uc?id=1uay-B3d0VGaoGhVEsVa6AqVn18z9yeyH -O sbert_vocab.txt
#gdown https://drive.google.com/uc?id=1LbzFUBBb3hD-dfl6qvYrBokJSmElZxKd -O sbert.graphdef
cd ../..
echo "Finish downloading! You can find models in static directory!"
