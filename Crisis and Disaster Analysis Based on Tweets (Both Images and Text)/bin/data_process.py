{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7bf71d70",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "Created on Sat Mar  4 20:42:24 2017\n",
    "@author: Firoj Alam\n",
    "\"\"\"\n",
    "\n",
    "\n",
    "import numpy as np\n",
    "np.random.seed(1337)  # for reproducibility\n",
    "\n",
    "import os\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "import sys\n",
    "from sklearn import preprocessing\n",
    "import pandas as pd\n",
    "import re\n",
    "from gensim.models import Word2Vec\n",
    "from gensim.models import KeyedVectors\n",
    "from nltk.corpus import stopwords\n",
    "import math\n",
    "stop_words = set(stopwords.words('english'))\n",
    "import random\n",
    "random.seed(1337)\n",
    "import aidrtokenize as aidrtokenize\n",
    "\n",
    "def file_exist(file_name):\n",
    "    if os.path.exists(file_name):\n",
    "        return True\n",
    "    else:\n",
    "        return False\n",
    "\n",
    "def read_stop_words(file_name):\n",
    "    if(not file_exist(file_name)):\n",
    "        print(\"Please check the file for stop words, it is not in provided location \"+file_name)\n",
    "        sys.exit(0)\n",
    "    stop_words =[]\n",
    "    with open(file_name, 'rU') as f:\n",
    "        for line in f:\n",
    "            line = line.strip()\n",
    "            if (line == \"\"):\n",
    "                continue\n",
    "            stop_words.append(line)\n",
    "    return stop_words;\n",
    "\n",
    "stop_words_file=\"stop_words_english.txt\"\n",
    "stop_words = read_stop_words(stop_words_file)\n",
    "\n",
    " \n",
    "\n",
    "def read_train_data(dataFile, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim):\n",
    "    \"\"\"\n",
    "    Prepare the data\n",
    "    \"\"\"\n",
    "    data=[]\n",
    "    lab=[]\n",
    "    with open(dataFile, 'rU') as f:\n",
    "        next(f)    \n",
    "        for line in f:\n",
    "            line = line.strip()   \n",
    "            if (line==\"\"):\n",
    "                continue                            \t\t\n",
    "            row=line.split(delim)\n",
    "            txt = row[3].strip()\n",
    "            txt = txt.replace(\"'\", \"\")\n",
    "            txt = aidrtokenize.tokenize(txt)\n",
    "\n",
    "            label = row[6]\n",
    "            txt = txt.replace(\"'\", \"\")\n",
    "            w_list=[]\n",
    "            for w in txt.split():\n",
    "                if w not in stop_words:\n",
    "                    try:\n",
    "                        #w=str(w.encode('ascii'))\n",
    "                        w_list.append(w.encode('utf-8'))\n",
    "                    except Exception as e:\n",
    "                        print(w)\n",
    "                        pass\n",
    "            text = \" \".join(w_list)\n",
    "\n",
    "            # if(len(text)<1):\n",
    "            #     print txt\n",
    "            #     continue\n",
    "            #txt=aidrtokenize.tokenize(txt)\n",
    "            #txt=[w for w in txt if w not in stop_words]              \n",
    "            if(isinstance(text, str)):\n",
    "                data.append(text)\n",
    "                lab.append(label)\n",
    "            else:\n",
    "                print(text)\n",
    "\n",
    "    data_shuf = []\n",
    "    lab_shuf = []\n",
    "    index_shuf = range(len(data))\n",
    "    random.shuffle(index_shuf)\n",
    "    for i in index_shuf:\n",
    "        data_shuf.append(data[i])\n",
    "        lab_shuf.append(lab[i])\n",
    "\n",
    "\n",
    "    le = preprocessing.LabelEncoder()\n",
    "    yL=le.fit_transform(lab_shuf)\n",
    "    labels=list(le.classes_)\n",
    "    \n",
    "    label=yL.tolist()\n",
    "    yC=len(set(label))\n",
    "    yR=len(label)\n",
    "    y = np.zeros((yR, yC))\n",
    "    y[np.arange(yR), yL] = 1\n",
    "    y=np.array(y,dtype=np.int32)\n",
    "    \n",
    "\n",
    "    # finally, vectorize the text samples into a 2D integer tensor\n",
    "    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token=\"OOV_TOK\")\n",
    "    tokenizer.fit_on_texts(data_shuf)\n",
    "    sequences = tokenizer.texts_to_sequences(data_shuf)\n",
    "    \n",
    "    word_index = tokenizer.word_index\n",
    "    print('Found %s unique tokens.' % len(word_index))\n",
    "    \n",
    "    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)\n",
    "    \n",
    "    #labels = to_categorical(np.asarray(labels))\n",
    "    print('Shape of data tensor:', data.shape)\n",
    "    #print('Shape of label tensor:', labels.shape)    \n",
    "    #return data,labels,word_index,dim;        \n",
    "    return data,y,le,labels,word_index,tokenizer\n",
    "\n",
    "    \n",
    "def read_dev_data(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):\n",
    "    \"\"\"\n",
    "    Prepare the data\n",
    "    \"\"\"      \n",
    "    data=[]\n",
    "    lab=[]\n",
    "    with open(dataFile, 'rU') as f:\n",
    "        next(f)    \n",
    "        for line in f:\n",
    "            line = line.strip()   \n",
    "            if (line==\"\"):\n",
    "                continue                            \t\t\n",
    "            row=line.split(delim)\n",
    "            txt = row[3].strip()\n",
    "            txt = txt.replace(\"'\", \"\")\n",
    "            txt = aidrtokenize.tokenize(txt)\n",
    "\n",
    "            label = row[6]\n",
    "\n",
    "            txt = txt.replace(\"'\",\"\")\n",
    "            w_list=[]\n",
    "            for w in txt.split():\n",
    "                if w not in stop_words:\n",
    "                    try:\n",
    "                        #w=str(w.encode('ascii'))\n",
    "                        w_list.append(w.encode('utf-8'))\n",
    "                    except Exception as e:\n",
    "                        #print(w)\n",
    "                        #print(e)\n",
    "                        pass\n",
    "            text = \" \".join(w_list)\n",
    "\n",
    "            # if(len(text)<1):\n",
    "            #     print txt\n",
    "            #     continue\n",
    "            #txt=aidrtokenize.tokenize(txt)\n",
    "            #txt=[w for w in txt if w not in stop_words]\n",
    "            if(isinstance(text, str)):\n",
    "                data.append(text)\n",
    "                lab.append(label)\n",
    "            else:\n",
    "                print(\"not text: \"+text)\n",
    "\n",
    "    le = train_le #preprocessing.LabelEncoder()\n",
    "    yL=le.transform(lab)\n",
    "    labels=list(le.classes_)\n",
    "    \n",
    "    label=yL.tolist()\n",
    "    yC=len(set(label))\n",
    "    yR=len(label)\n",
    "    y = np.zeros((yR, yC))\n",
    "    y[np.arange(yR), yL] = 1\n",
    "    y=np.array(y,dtype=np.int32)\n",
    "\n",
    "    sequences = tokenizer.texts_to_sequences(data)   \n",
    "    word_index = tokenizer.word_index\n",
    "    print('Found %s unique tokens.' % len(word_index))    \n",
    "    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)\n",
    "    print('Shape of data tensor:', data.shape)\n",
    "    return data,y,le,labels,word_index\n",
    "\n",
    "\n",
    "def read_data_classifier(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):\n",
    "    \"\"\"\n",
    "    Prepare the data\n",
    "    \"\"\"\n",
    "    data = []\n",
    "    lab = []\n",
    "    with open(dataFile, 'rU') as f:\n",
    "        next(f)\n",
    "        for line in f:\n",
    "            line = line.strip()\n",
    "            if (line == \"\"):\n",
    "                continue\n",
    "            row = line.split(delim)\n",
    "            txt = row[3].strip()\n",
    "            txt = txt.replace(\"'\", \"\")\n",
    "            txt = aidrtokenize.tokenize(txt)\n",
    "\n",
    "            label = row[6]\n",
    "\n",
    "            txt = txt.replace(\"'\", \"\")\n",
    "            w_list = []\n",
    "            for w in txt.split():\n",
    "                if w not in stop_words:\n",
    "                    try:\n",
    "                        # w=str(w.encode('ascii'))\n",
    "                        w_list.append(w.encode('utf-8'))\n",
    "                    except Exception as e:\n",
    "                        # print(w)\n",
    "                        # print(e)\n",
    "                        pass\n",
    "            text = \" \".join(w_list)\n",
    "\n",
    "            # if(len(text)<1):\n",
    "            #     print txt\n",
    "            #     continue\n",
    "            # txt=aidrtokenize.tokenize(txt)\n",
    "            # txt=[w for w in txt if w not in stop_words]\n",
    "            if (isinstance(text, str)):\n",
    "                data.append(text)\n",
    "                lab.append(label)\n",
    "            else:\n",
    "                print(\"not text: \" + text)\n",
    "\n",
    "    # le = train_le  # preprocessing.LabelEncoder()\n",
    "    # yL = le.transform(lab)\n",
    "    # labels = list(le.classes_)\n",
    "    #\n",
    "    # label = yL.tolist()\n",
    "    # yC = len(set(label))\n",
    "    # yR = len(label)\n",
    "    # y = np.zeros((yR, yC))\n",
    "    # y[np.arange(yR), yL] = 1\n",
    "    # y = np.array(y, dtype=np.int32)\n",
    "\n",
    "    sequences = tokenizer.texts_to_sequences(data)\n",
    "    word_index = tokenizer.word_index\n",
    "    print('Found %s unique tokens.' % len(word_index))\n",
    "    data_x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)\n",
    "    print('Shape of data tensor:', data_x.shape)\n",
    "    return data_x, data, lab, #le, labels, word_index\n",
    "\n",
    "def load_embedding(fileName):\n",
    "    print('Indexing word vectors.')    \n",
    "    embeddings_index = {}    \n",
    "    f = open(fileName)\n",
    "    for line in f:\n",
    "        values = line.split()\n",
    "        word = values[0]\n",
    "        coefs = np.asarray(values[1:], dtype='float32')\n",
    "        embeddings_index[word] = coefs\n",
    "    f.close()    \n",
    "    print('Found %s word vectors.' % len(embeddings_index))\n",
    "    return embeddings_index;\n",
    "\n",
    "def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):\n",
    "    \n",
    "    # prepare embedding matrix\n",
    "    nb_words = min(MAX_NB_WORDS, len(word_index)+1)    \n",
    "    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM),dtype=np.float32)\n",
    "    print(len(embedding_matrix))\n",
    "    for word, i in word_index.items():\n",
    "        if i >= nb_words:\n",
    "            continue\n",
    "        try:\n",
    "            embedding_vector = model[word][0:EMBEDDING_DIM] #embeddings_index.get(word)\n",
    "            embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)\n",
    "        except KeyError:\n",
    "            try:\n",
    "                print(word +\" not found... assigning zeros\")\n",
    "                rng = np.random.RandomState()        \t\n",
    "                #embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)\n",
    "                embedding_vector = np.zeros(EMBEDDING_DIM)  # np.random.random(num_features)\n",
    "                embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)\n",
    "            except KeyError:    \n",
    "                continue      \n",
    "    return embedding_matrix;\n",
    "\n",
    "def str_to_indexes(s):\n",
    "    alphabet = \"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\\\"/\\\\|_@#$%^&*~`+-=<>()[]{}\"\n",
    "    input_size = 1014\n",
    "    length = input_size\n",
    "    alphabet_size = len(alphabet)\n",
    "    char_dict = {}  # Maps each character to an integer\n",
    "    # self.no_of_classes = num_of_classes\n",
    "    for idx, char in enumerate(alphabet):\n",
    "        char_dict[char] = idx + 1\n",
    "    length = input_size\n",
    "\n",
    "\n",
    "    \"\"\"\n",
    "    Convert a string to character indexes based on character dictionary.\n",
    "    Args:\n",
    "        s (str): String to be converted to indexes\n",
    "    Returns:\n",
    "        str2idx (np.ndarray): Indexes of characters in s\n",
    "    \"\"\"\n",
    "    s = s.lower()\n",
    "    max_length = min(len(s), length)\n",
    "    str2idx = np.zeros(length, dtype='int64')\n",
    "    for i in range(1, max_length + 1):\n",
    "        c = s[-i]\n",
    "        if c in char_dict:\n",
    "            str2idx[i - 1] = char_dict[c]\n",
    "    return str2idx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e92a079",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
