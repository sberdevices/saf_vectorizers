echo "## Creating static dir for models..."
cd saf_vectorizers
mkdir -p static
cd static

for model_name in "$@"
do
    if [[ "$model_name" == *"sbert"* || "$model_name" == "all" ]] && [ ! -d sbert ]; then
      echo "## Downloading SBERT model..."
      mkdir -p sbert
      cd sbert
      curl -L https://sc.link/A51 > sbert_vocab.txt
      curl -L https://sc.link/zG7 > sbert.graphdef
      cd ..
    fi

    if [[ "$model_name" == *"use"* || "$model_name" == "all" ]] && [ ! -d universal_sentence_encoder ]; then
      echo "## Downloading USE embeddings and unpacking it..."
      curl -L https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed > universal_sentence_encoder.tar.gz
      mkdir -p universal_sentence_encoder
      tar -xvf universal_sentence_encoder.tar.gz -C universal_sentence_encoder
    fi

    if [[ "$model_name" == *"word2vec"* || "$model_name" == "all" ]] && [ ! -d word2vec ]; then
      echo "## Downloading Word2Vec embeddings and unpacking it..."
      curl -L http://vectors.nlpl.eu/repository/20/182.zip > word2vec.zip
      mkdir -p word2vec
      unzip word2vec.zip -d word2vec
    fi

    if [[ "$model_name" == *"fasttext"* || "$model_name" == "all" ]] && [ ! -d fasttext ]; then
      echo "## Downloading FastText embeddings and unpacking it..."
      curl -L https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz > fasttext.bin.gz
      mkdir -p fasttext
      gunzip -c fasttext.bin.gz > ./fasttext/fasttext.bin
    fi
done

echo "## Static dir contains the following files:"
echo $(ls)
echo "## Finish!"
