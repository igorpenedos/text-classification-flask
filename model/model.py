import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class Model:
    
    dictionary = {
        0: "Politics",
        1: "Sport",
        2: "Technology",
        3: "Entertainment",
        4: "Business"
    }
    clf = None
    cv = CountVectorizer()
    accuracy = 0
    
    def init(self):
        data = pd.read_csv("./data/data.csv", index_col=False)
        X = data["Text"]
        Y = data["Label"]
        
        X = self.cv.fit_transform(X)
        
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=13)
        
        self.clf = MultinomialNB()
        self.clf.fit(trainX, trainY)
        
        yPred = self.clf.predict(testX)
        
        self.accuracy = accuracy_score(testY, yPred)
        
    def predict(self, text):
        text_to_predict = self.cv.transform([text])
        label = self.clf.predict(text_to_predict)[0]
        result = self.dictionary[label]
        return result