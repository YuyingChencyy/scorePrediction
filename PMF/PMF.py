# -*- coding: utf-8 -*-
import sys
import numpy as np

def my_norm(matrix_, t=2):
    # calculte sum of the squared errors(SSE)/2

    sum_ = matrix_ ** 2
    sum_ = sum_.sum()
    return sum_ / t

def random_split(percentage=0.3):
    # split train data and test data
    rating_all = np.loadtxt(r"rating.txt", delimiter="\t")
    test_index = np.random.choice(np.arange(0, len(rating_all)), int(np.around(percentage * len(rating_all))),
                                  replace=False)
    train_index = list(set(np.arange(0, len(rating_all))).difference(set(test_index)))
    train = list()
    for i in train_index:
        train.append(rating_all[i])
    train = np.array(train)

    test = list()
    for i in test_index:
        test.append(rating_all[i])
    test = np.array(test)
    return train, test

def factorization(argv):
    M=int(argv[0])
    N=int(argv[1])
    dimension=int(argv[2])
    alpha=float(argv[3])
    lamda=float(argv[4])
    percentage=float(argv[5])

    train,test= random_split(percentage)
    mu, sigma = 0, 1
    P=0.1*np.random.normal(mu, sigma, size=(M, dimension))
    Q=0.1*np.random.normal(mu, sigma, size=(N, dimension))
    cost_func_value_old = 0
    steps = 1000
    maxbatch=10

    for step in np.arange(steps):
        print("PMF")
        print("step= ",step)
        user=[int(t[0]) for t in train]
        item=[int(t[1]) for t in train]
        rating=np.array([float(t[2]) for t in train])

        predict=np.array([np.dot(P[user[i]],Q[item[i]]) for i in np.arange(len(user))])

        cost_func_value = sum(np.square(predict-rating)) + lamda * (my_norm(Q, 2) + my_norm(P, 2))
        print("cost_func_value= ",cost_func_value)
        for batch in np.arange(maxbatch):
            print("this is the ", batch, " batch")
            batch_index=np.random.choice(np.arange(0, len(train)), int(np.around((1.0/maxbatch) * len(train))),replace=False)

            u_gradient = np.zeros((M, dimension))
            v_gradient = np.zeros((N, dimension))
            # U gradient update and V gradient
            for i in batch_index:
                user_index=int(train[i][0])
                item_index=int(train[i][1])
                rating=float(train[i][2])
                temp=np.dot(P[user_index],Q[item_index])-rating
                u_gradient[user_index]+=temp*Q[item_index]
                v_gradient[item_index]+=temp*P[user_index]
            P-= alpha*(u_gradient + lamda * P)
            Q-= alpha*(v_gradient + lamda * Q)
        if step>20:
            if abs(cost_func_value- cost_func_value_old ) < 10:
                print('Program exit after convergence!')
                user = [int(t[0]) for t in test]
                item = [int(t[1]) for t in test]
                rating =np.array([float(t[2]) for t in test])
                predict =np.array([np.dot(P[user[i]], Q[item[i]]) for i in np.arange(len(user))])
                for i in range(len(predict)):
                    if predict[i]>1:
                        predict[i]=1
                    if predict[i]<0:
                        predict[i]=0
                    test_set_error =predict - rating
                    mae_test= np.sum(abs(test_set_error)) / (len(test_set_error))
                    rmse_test= np.sum(pow(test_set_error, 2)) /(len(test_set_error))
                    rmse_test= np.sqrt(rmse_test)
                # ------------diversity test----------
                print("test_mae:", mae_test, "test_rmse:", rmse_test)
                break
            else:
                cost_func_value_old = cost_func_value
        else:
            cost_func_value_old = cost_func_value

        if step ==steps-1:
            user = [int(t[0]) for t in test]
            item = [int(t[1]) for t in test]
            rating =np.array([float(t[2]) for t in test])
            predict =np.array([np.dot(P[user[i]], Q[item[i]]) for i in np.arange(len(user))])
            for i in range(len(predict)):
                if predict[i]>1:
                    predict[i]=1
                if predict[i]<0:
                    predict[i]=0
            test_set_error =predict - rating
            mae_test= np.sum(abs(test_set_error)) / (len(test_set_error))
            rmse_test= np.sum(pow(test_set_error, 2)) /(len(test_set_error))
            rmse_test= np.sqrt(rmse_test)
            # ------------diversity test----------
            print("test_mae:", mae_test, "test_rmse:", rmse_test)

print("***************")

if __name__ == "__main__":
    factorization(sys.argv[1:])


