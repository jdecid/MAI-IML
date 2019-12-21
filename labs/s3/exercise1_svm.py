#############################################################
#############################################################
#############################################################


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

if __name__ == "__main__":
    def generate_data_set1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def generate_data_set2():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def generate_data_set3():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train


    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test


    def plot_data(svc, X_train, y_train, X_test, y_test, title):
        h = 0.01

        x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1, \
                       max(X_train[:, 0].max(), X_test[:, 0].max()) + 1

        y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1, \
                       max(X_train[:, 1].max(), X_test[:, 1].max()) + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, cmap=plt.cm.coolwarm)
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test, cmap=plt.cm.coolwarm)
        plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], marker='+', c='black')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(f'{title} with train data (o) and test data (x)')

        plt.show()


    def print_accuracies(y, y_pred, disp=True):
        corrects = sum(y_pred == y)
        accuracy = 100 * corrects / len(y)
        if disp:
            print(f'Correct results: {corrects}/{len(y)} ({accuracy}%)')
        return accuracy


    def run_svm_dataset1():
        X1, y1, X2, y2 = generate_data_set1()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        svc = SVC(kernel='linear', C=100, gamma='auto')
        svc.fit(X_train, y_train)
        results = svc.predict(X_test)

        plot_data(svc, X_train, y_train, X_test, y_test, 'Linear kernel')
        print_accuracies(y_test, results)


    def run_svm_dataset2():
        X1, y1, X2, y2 = generate_data_set2()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        best_acc = 1
        best_c = None

        for C in [0.01, 0.1, 1, 10, 100]:
            cval_accuracies = []
            for i in range(10):
                X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.2)
                svc = SVC(kernel='linear', C=C, gamma='auto')
                svc.fit(X_t, y_t)
                results = svc.predict(X_val)
                accuracy = print_accuracies(y_val, results, disp=False)
                cval_accuracies.append(accuracy)

            accuracy = np.mean(cval_accuracies)

            if accuracy > best_acc:
                best_acc = accuracy
                best_c = C

        svc = SVC(kernel='linear', C=best_c, gamma='auto')
        svc.fit(X_train, y_train)
        results = svc.predict(X_test)

        plot_data(svc, X_train, y_train, X_test, y_test, f'Linear kernel (C={best_c})')
        print_accuracies(y_test, results)


    def run_svm_dataset3():
        X1, y1, X2, y2 = generate_data_set3()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        svc = SVC(kernel='rbf', C=100, gamma=0.7)
        svc.fit(X_train, y_train)
        results = svc.predict(X_test)

        plot_data(svc, X_train, y_train, X_test, y_test, 'Linear kernel')
        print_accuracies(y_test, results)

        #### 
        # Write here your SVM code and use a gaussian kernel 
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####


    #############################################################
    #############################################################
    #############################################################

    # EXECUTE SVM with THIS DATASETS
    run_svm_dataset1()  # data distribution 1
    run_svm_dataset2()  # data distribution 2
    run_svm_dataset3()  # data distribution 3

#############################################################
#############################################################
#############################################################
