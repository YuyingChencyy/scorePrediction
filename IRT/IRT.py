# -*- coding: utf-8 -*-
import sys
import numpy as np
from math import exp, log

def random_split(percentage=0.3):
    # split train data and test data
    rating_all = np.loadtxt(r"rating.txt", delimiter="\t")
    test_index = np.random.choice(np.arange(0, len(rating_all)), int(np.around(percentage * len(rating_all))),replace=False)
    train_index = list(set(np.arange(0, len(rating_all))).difference(set(test_index)))
    train=list()
    for i in train_index:
        train.append(rating_all[i])
    train=np.array(train)

    test=list()
    for i in test_index:
        test.append(rating_all[i])
    test=np.array(test)
    return train,test

def train_model(argv):
    M=int(argv[0])
    N=int(argv[1])
    alpha=float(argv[2])
    percentage=float(argv[3])
    train,test= random_split(percentage)
    A= np.random.rand(N)
    B= np.random.rand(N)
    D=1.7
    Theta= np.random.rand(M)
    likelihood_value_old= 0
    steps = 1000
    maxbatch=10

    for step in np.arange(steps):
        print("IRT")
        print("step= ",step)
        user=[int(t[0]) for t in train]
        item=[int(t[1]) for t in train]
        rating=np.array([float(t[2]) for t in train])
        predict=np.array([(1.0/(1.0+exp(-D*A[item[i]]*(Theta[user[i]]-B[item[i]])))) for i in np.arange(len(user))])
        likelihood_value =sum([(rating[i]* log(predict[i])+(1.0-rating[i])*log(1.001-predict[i])) for i in range(len(predict))])
        print("likelihood_value = ",likelihood_value )
        for batch in np.arange(maxbatch):
            print("this is the ", batch, " batch")
            batch_index=np.random.choice(np.arange(0, len(train)), int(np.around((1.0/maxbatch) * len(train))),replace=False)
            a_gradient = np.zeros(N)
            b_gradient = np.zeros(N)
            theta_gradient = np.zeros(M)
            # A 、B and Theta gradient update
            for i in batch_index:
                user_index=int(train[i][0])
                item_index=int(train[i][1])
                rating=float(train[i][2])
                temp=rating-1.0/(1.0+exp(-D*A[item_index]*(Theta[user_index]-B[item_index])))
                a_gradient[item_index]+=temp*D
                b_gradient[item_index]+=-temp*D*A[item_index]
                theta_gradient[user_index]=temp*D*A[item_index]
            A+= alpha*a_gradient
            B+= alpha*b_gradient
            Theta+=alpha*theta_gradient
        if step>20:
            if abs(likelihood_value- likelihood_value_old ) < 1000:
                print('Program exit after convergence!')
                user = [int(t[0]) for t in test]
                item = [int(t[1]) for t in test]
                rating =np.array([float(t[2]) for t in test])
                predict =np.array([(1.0/(1.0+exp(-D*A[item[i]]*(Theta[user[i]]-B[item[i]])))) for i in np.arange(len(user))])
                test_set_error =predict - rating
                mae_test= np.sum(abs(test_set_error)) / (len(test_set_error))
                rmse_test= np.sum(pow(test_set_error, 2)) /(len(test_set_error))
                rmse_test= np.sqrt(rmse_test)
                # ------------diversity test----------
                print("test_mae:", mae_test, "test_rmse:", rmse_test)
                break
            else:
                likelihood_value_old = likelihood_value
        else:
            likelihood_value_old = likelihood_value

        if step == steps-1:
            user = [int(t[0]) for t in test]
            item = [int(t[1]) for t in test]
            rating =np.array([float(t[2]) for t in test])
            predict =np.array([(1.0/(1.0+exp(-D*A[item[i]]*(Theta[user[i]]-B[item[i]])))) for i in np.arange(len(user))])
            test_set_error =predict - rating
            mae_test= np.sum(abs(test_set_error)) / (len(test_set_error))
            rmse_test= np.sum(pow(test_set_error, 2)) /(len(test_set_error))
            rmse_test= np.sqrt(rmse_test)
            # ------------diversity test----------
            print("test_mae:", mae_test, "test_rmse:", rmse_test)

print("***************")

#factorization(M, N,  alpha,percentage)
#train_model(3217,411,0.001,0.3)

if __name__ == "__main__":
    train_model(sys.argv[1:])