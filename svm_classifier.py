from sklearn import svm

X = [[0, 1], [1, 1]]

y = [0, 1]

clf = svm.SVC()

clf.fit(X, y)

pred = clf.predict([[2., 2.], [0, 0]])

for p in pred:
    print p
