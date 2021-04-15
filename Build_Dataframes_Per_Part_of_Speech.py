from google.colab import files

file = files.upload()

print(len(file.keys()))

print(file)

import pandas

import io 

dataframe = pandas.read_csv(io.StringIO(file["author_and_quote.csv"].decode("utf-8")), 
                            sep=",")

print(dataframe.head().to_string())

print(dataframe.head())

print(dataframe.describe())

quotes_list = dataframe["Quote"].tolist()

print(quotes_list)

print(len(quotes_list))

import nltk
import unicodedata
import string
import re

def remove_accents(token):
  return "".join(x for x in unicodedata.normalize("NFKD", token) if x in string.ascii_letters or x == " ")

nltk.download("stopwords")

nltk.download("punkt")

nltk.download("averaged_perceptron_tagger")

nltk.download("wordnet")

stopwords  = nltk.corpus.stopwords.words("english")

stemmer    = nltk.stem.PorterStemmer()

lemmatizer = nltk.stem.WordNetLemmatizer()

RE_VALID = "[a-zA-Z]"
MINIMUM_STRING_LENGTH = 3

ALLOWED_PART_OF_SPEECH_TYPES = {"NN": "n", 
                                "JJ":"a", 
                                "VB":"v",
                                "RB":"r"}

PART_OF_SPEECH_TYPES_KEYS = list(ALLOWED_PART_OF_SPEECH_TYPES.keys())

tokens_list           = []

all_tokens_lists      = []

all_lemmatized_tokens = []

for index, text in enumerate(quotes_list):

  tokens = [word.lower() for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]

  list_sentence_tokens  = []
  non_lemmatized_tokens = []

  for token in tokens:

    result = remove_accents(token)

    result = str(result).translate(string.punctuation)

    list_sentence_tokens.append(result)

    non_lemmatized_tokens.append("-")

    if result not in stopwords:

      if re.search(RE_VALID, result):

        if len(result) >= MINIMUM_STRING_LENGTH:

          part_of_speech = nltk.pos_tag([result])[0][1][:2]

          default_part_of_speech = "n"

          if part_of_speech in ALLOWED_PART_OF_SPEECH_TYPES:

            default_part_of_speech = ALLOWED_PART_OF_SPEECH_TYPES[part_of_speech]

          stem = stemmer.stem(result)

          lemmatization = lemmatizer.lemmatize(result,
                                               pos=default_part_of_speech)
          
          if part_of_speech in PART_OF_SPEECH_TYPES_KEYS:

            tokens_list.append((result,
                                stem,
                                lemmatization,
                                part_of_speech))
            
            non_lemmatized_tokens = non_lemmatized_tokens[:-1]

            non_lemmatized_tokens.append(lemmatization)

  all_tokens_lists.append(list_sentence_tokens)

  lemmatized_tokens_list = " ".join(non_lemmatized_tokens)

  all_lemmatized_tokens.append(lemmatized_tokens_list)

dataframe_tokens = pandas.DataFrame(all_tokens_lists)

print(dataframe_tokens)

print(dataframe_tokens.head().to_string())

for entry in dataframe_tokens:

  if str(dataframe_tokens[entry].dtype) in ("object", "string_", "unicode_"):

    dataframe_tokens[entry].fillna(value="", inplace=True)

print(dataframe_tokens.head().to_string())

dataframe_all_words = pandas.DataFrame(tokens_list, columns=["token",
                                                            "stem",
                                                            "lemmatization",
                                                            "part_of_speech"])

print(dataframe_all_words.to_string())

dataframe_all_words["counts"] = dataframe_all_words.groupby(["lemmatization"])["lemmatization"].transform("count")

print(dataframe_all_words["counts"])

print(dataframe_all_words)

dataframe_all_words = dataframe_all_words.sort_values(by=["counts", 
                                                          "lemmatization"], 
                                        ascending=[False, True]).reset_index()

print(dataframe_all_words.to_string())

dataframe_grouped = dataframe_all_words.groupby("lemmatization").first().sort_values(by="counts",
                                                                 ascending=False).reset_index()

print(dataframe_grouped.to_string())

dataframe_grouped = dataframe_grouped[["lemmatization", "part_of_speech", "counts"]]

for part_of_speech_type in PART_OF_SPEECH_TYPES_KEYS: 

  dataframe_part_of_speech = dataframe_grouped[dataframe_grouped["part_of_speech"] == part_of_speech_type]

  print(dataframe_part_of_speech.to_string())
