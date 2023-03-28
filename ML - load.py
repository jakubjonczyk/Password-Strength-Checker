import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed
import joblib
  
# Load the model from the file
model_from_joblib = joblib.load('savefile.pkl')

data = pd.read_csv("Dane.csv", error_bad_lines=False)

data = data.dropna()
data["difficulty"] = data["difficulty"].map({0: "Easy", 
                                         1: "Medium",
                                         2: "Hard"})

#preparing for using loaded model
def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  
x = np.array(data["password"])
y = np.array(data["difficulty"])

tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.05, 
                                                random_state=42)
  
# Use the loaded model
import getpass
user = getpass.getpass("Enter Password: ")
data = tdif.transform([user]).toarray()
output = model_from_joblib.predict(data)
print(output)