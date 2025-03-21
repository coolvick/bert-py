import tensorflow as tf
print(tf.__version__)

import os

# file_path = 'C:\\Users\\hashe\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\tensorflow_decision_forests\\tensorflow\\ops\\inference\\inference.so'
file_path = 'C:\\tryouts\\bert-py\\.venv\\Lib\\site-packages\\tensorflow_decision_forests\\tensorflow\\ops\\inference\\inference.so'
print(os.path.isfile(file_path)) 

# inference_so_dir = 'C:\\Users\\hashe\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\tensorflow_decision_forests\\tensorflow\\ops\\inference'
inference_so_dir = 'C:\\tryouts\\bert-py\\.venv\\Lib\\site-packages\\tensorflow_decision_forests\\tensorflow\\ops\\inference'
os.environ['PATH'] = inference_so_dir + os.pathsep + os.environ['PATH']

import tensorflow_decision_forests as tfdf