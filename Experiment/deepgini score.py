import pandas as pd
import math

def get_deepgini_score(path):
    header = ['filename', 'probability']
    df = pd.read_csv(path, names=header)
    Sum = 0
    for i in range(len(df)):
        probability = df['probability'][i].strip('[]').split(",")
        sum = 0
        for j in range(len(probability)):
            p = float(probability[j])
            sum += math.pow(p, 2)
        each_score = 1 - sum
        print(df['filename'][i]+"_score: " + str(each_score))
        Sum += each_score
    deepgini_score = Sum/len(df)
    print(deepgini_score)

get_deepgini_score('./submit/resnet50_Aug_1_submission.csv')