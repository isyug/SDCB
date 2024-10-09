from predictor import Predictor
import tools
import numpy as np
import pandas as pd

RUNS = 10
FLODNUMS = 5
datas = tools.data(FLODNUMS)
for t in range(RUNS):
    datas.split()
    for i in range(FLODNUMS):
        train, test = datas.getFitData(t, i)

        model = Predictor(t, i)
        model.fit(train[:, :-1], train[:, -1])

        label = test[:, -1]
        preSVM = model.svm.predict_proba(test[:, 2:-1])[:, 1]
        preBernNb = model.bernoulliNB.predict_proba(test[:, 2:-1])[:, 1]
        preTL = model.tl_predict(test[:, 0:2])

        svms = np.concatenate((svms, preSVM), axis=0)
        bernnb = np.concatenate((bernnb, preBernNb), axis=0)
        tls = np.concatenate((tls, preTL), axis=0)
        labels = np.concatenate((labels, label), axis=0)

pd.DataFrame(svms, columns=["svm"]).to_csv("./svm.csv".format(t), index=False)
pd.DataFrame(bernnb, columns=["bernnb"]).to_csv("./bernnb.csv".format(t), index=False)
pd.DataFrame(tls, columns=["tl"]).to_csv("./tl.csv".format(t), index=False)
pd.DataFrame(labels, columns=["label"]).to_csv("./label.csv".format(t), index=False)
