import torch
from torch import nn

def f(x, w, b):
    return torch.matmul(x, w) + b

def MSE(yhat, y):
    p = y.size(1)
    loss = p * ((yhat - y)**2).mean()
    return loss

def reglin(x, y, xtest, ytest, batchsize = 32, eps = 1e-3, n_iter = 100):
    # créer les paramètres w, b     avec  require.grad = True
    w = torch.randn(size=(x.size(1),x.size(0)), requires_grad=True)            # size=(n,p) 矩阵
    b = torch.randn(size=(1,y.size(1)), requires_grad=True)                    # size=(1,p) 行向量

    # boucle sur le nombre d'epoch (n_iter)
    for epoch in range(n_iter):                                                        

        # boucle sur le parcours du dataset en mini-batch
        for i in range(0, len(x), batchsize):
            x_batch = x[i:i+batchsize]
            y_batch = y[i:i+batchsize]

            # Forward: Calcul de la prédiction 
            yhat = f(x_batch, w, b)

            # Calcul de la perte (MSE)
            loss = MSE(yhat, y_batch)
            
            # Backward: calcul des gradients
            loss.backward()

            # mise à jour paramètres de w, b sans calcul des gradients
            with torch.no_grad():
                w -= eps * w.grad
                b -= eps * b.grad

            # remise à zéro des gradients
            w.grad.zero_()
            b.grad.zero_()

    # inférence with no.grad
    with torch.no_grad():
        yhat_test = f(xtest, w, b)
        test_loss = MSE(yhat_test, ytest)
        print(f"Test loss : {test_loss.item()}")

    return 0