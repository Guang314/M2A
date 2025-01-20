import torch
from torch.utils.tensorboard import SummaryWriter
from tp2 import MSE, Linear, Context

# Les données supervisées
x = torch.randn(50, 13, requires_grad=False)
y = torch.randn(50, 3, requires_grad=False)

# Les paramètres du modèle à optimiser
w = torch.nn.Parameter(torch.randn(13,3))
b = torch.nn.Parameter(torch.randn(3))

epsilon = 0.05
nb_epoch = 400

# On optimise selon w et b, lr: pas de gradient
optim = torch.optim.Adam(params=[w,b], lr=epsilon)    # Adam 优化算法
optim.zero_grad()

# Reinitialisation du gradient
writer = SummaryWriter()
for n_iter in range(nb_epoch):

    # Calcul du forward (loss)
    yhat = Linear.apply(x,w,b)
    loss = MSE.apply(yhat,y)

    # tensorboard --logdir runs/
    writer.add_scalar('Adam Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    # Retropropagation
    loss.backward()

    # if n_iter % 10 == 0:
    #     optim.step()
    #     optim.zero_grad()

    optim.step()
    optim.zero_grad()

writer.close()
