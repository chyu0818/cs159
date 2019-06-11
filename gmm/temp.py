import numpy as np
import sklearn
import pickle
from sklearn.mixture import GaussianMixture


gmm = GaussianMixture(n_components=30)


X = []
y = []
with open("../embeddings/word_vec_tfidf1.txt") as f:
    newlines = f.readlines()
    for x in newlines:
        if x == "\n":
            continue
        
        real_thing = x.split("]")
        
        hehe = real_thing[0].split(" ")
        emotion = hehe[len(hehe) - 1]
        arr = (real_thing[0].replace(emotion, "") + "]").replace("[", "").replace("]", "")
        new_arr = np.array(arr.split(","))
        
        final_arr = []
        for temp_str in new_arr:
            if (temp_str != " "):
                final_arr.append(float(temp_str.replace(",", "")))
        
        X.append(final_arr)
        y.append(emotion)

def save_model(file_name, model):
    pkl_filename = file_name  
    with open(file_name, 'wb') as file:  
        pickle.dump(model, file)
        
def get_model(file_name):
    with open(file_name, 'rb') as file:
        temp_model = pickle.load(file)
        return temp_model


gmm.fit(X)

save_model("final_30_emotions_gmm.pkl", gmm)