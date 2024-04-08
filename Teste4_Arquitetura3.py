import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

arquivo = np.load('teste4.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

regr = MLPRegressor(hidden_layer_sizes=(100, 100),
                    max_iter=200000,
                    activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=500)

regr = regr.fit(x,y)
y_est = regr.predict(x)
erro = np.abs(y_est - y) #Erro absoluto

print("Desvio Padrão: ", np.std(erro)) #Desvio Padrão do Erro
print("Média: ", np.mean(erro)) #Média do Erro

plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr.loss_curve_)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,y_est,linewidth=2)

plt.show()
