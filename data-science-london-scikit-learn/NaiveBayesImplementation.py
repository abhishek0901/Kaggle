train_data = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv")
test_data = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv")
train_lebels = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv")

train_data_norm =(train_data-train_data.mean())/train_data.std()

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(train_data_norm.to_numpy(), train_lebels.to_numpy().reshape(-1)).predict(test_data_norm.to_numpy())