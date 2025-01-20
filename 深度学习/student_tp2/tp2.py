
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors
    
    
class Linear(Function):
    """Début d'implementation de la fonction Linear(X, W, b)"""
    @staticmethod
    def forward(ctx, X, W, b):
        # Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W, b)

        # Renvoyer la valeur de la fonction Linear
        Y_hat = X @ W + b

        return Y_hat 
    
    @staticmethod
    def backward(ctx, grad_outputs):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        X, W, b = ctx.saved_tensors
        
        # Renvoyer par les deux dérivées partielles (par rapport à X, W, b)
        grad_X = grad_outputs @ W.T
        grad_W = X.T @ grad_outputs
        grad_b = grad_outputs.sum(dim=0)

        return grad_X, grad_W, grad_b


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        loss = ((yhat - y)**2).mean()
        return loss
        

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        n_q = y.size(0)

        grad_yhat = 2 * (yhat - y) / n_q
        grad_y = 2 * (y - yhat) / n_q

        grad_yhat *= grad_output
        grad_y *= grad_output

        return grad_yhat, grad_y


## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

