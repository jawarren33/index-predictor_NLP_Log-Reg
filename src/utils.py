#pip install transformers

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import random
import torch
import tensorflow_hub as hub
import tensorflow_text as text
from matplotlib import pyplot
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from sklearn.naive_bayes import GaussianNB
