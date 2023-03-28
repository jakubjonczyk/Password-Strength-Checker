import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# read data from file
data = pd.read_csv("Dane.csv", error_bad_lines=False)

data = data.dropna()
data["difficulty"] = data["difficulty"].map({0: "Easy", 
                                         1: "Medium",
                                         2: "Hard"})

#preparing for training model
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
#training model
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

from joblib import Parallel, delayed
import joblib
  
# Save the model
joblib.dump(model, 'savefile.pkl')