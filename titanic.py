# Author: Chromilo Amin  
# Date: Dec 29, 2020
# Description: Code below is primarily from https://www.tensorflow.org/tutorials/load_data/csv to predict given a set of test data who survives and who doesn't when the Titanic goes down
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

titanic_train = pd.read_csv("/home/chromilo/train.csv")
titanic_train.head()

titanic_features = titanic_train.copy()
titanic_labels = titanic_features.pop('Survived')
# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

inputs = {}
for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
inputs

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}
                 
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(titanic_features[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue


  lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic_features[name]))
  print(lookup(input))
  one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

  
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)



titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}

features_dict = {name:values for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)


def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64,activation='sigmoid'),
    layers.Dense(1,activation='sigmoid')
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  #model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                   optimizer=tf.optimizers.Adam())
  return model

titanic_model = titanic_model(titanic_preprocessing, inputs)
x=titanic_features_dict
y=titanic_labels
titanic_model.fit(x, y, epochs=100)
#----------------------------
titanic_test = pd.read_csv("/home/chromilo/test.csv")
titanic_test.head()
titanic_features_test = titanic_test.copy()
#titanic_labels = titanic_features.pop('Survived')
titanic_features_dict_test = {name: np.array(value) 
                         for name, value in titanic_features_test.items()}
x_test=titanic_features_dict_test
#y_test=titanic_features_dict   

# Evaluate the model on the test data using `evaluate`
#print("Evaluate on test data")
#results = titanic_model.evaluate(x_test, y_test, batch_size=128)
#print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
predictions = titanic_model.predict(x_test)
#predictions = np.argmax(predictions, axis = 1)
predictions = np.round(predictions)

n=0
survived=0
submission = []
for u in predictions:
   dta = []
   passenger = int(titanic_test.loc[n][0])
   dta.append(passenger)
   if(u>0):
      survived += 1
      dta.append("1")
   else:
      dta.append("0")  
   submission.append(dta)
   n += 1    
df = pd.DataFrame(submission) 
df.columns = ['PassengerId', 'Survived']
print(df)
csv_data = df.to_csv('/home/chromilo/submission.csv', index = False) 
print(n,survived)

tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)
