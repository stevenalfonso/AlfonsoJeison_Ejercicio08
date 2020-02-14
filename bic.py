import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("data_to_fit.txt", skiprows=1)
X = data[:,0]
Y = data[:,1]
sigma_y = data[:,2]

def model_A(x, params):
    y = params[0] + params[1] * x + params[2] * x**2
    return y

def model_B(x, params):
    y = params[0]*(np.exp(-0.5*(x - params[1])**2 / params[2]**2))
    return y

def model_C(x, params):
    y = params[0] * (np.exp(-0.5*(x - params[1])**2/params[2]**2))
    y += params[0] * (np.exp(-0.5*(x - params[3])**2/params[4]**2))
    return y

def loglike(x, y, sigma_y, params, model):
    L = 0.0
    for i in range(len(y)):
        L += -0.5 * (y[i] - model(x[i], params))**2/sigma_y[i]**2
    return L

def metropolis(N, model, params):
    #params = np.ones(3)
    p = loglike(X, Y, sigma_y, params, model)
    new_params = np.zeros((N, len(params)))
    for i in range(N):
        params_n = params + np.random.normal(0, 0.05, len(params))
        p_n = loglike(X, Y, sigma_y, params_n, model)
        u = np.random.rand()
        if u < min(1, np.exp(p_n - p)):
            p = p_n
            params = params_n
        new_params[i] = params
    return np.array(new_params)

N = 50000
params_A = metropolis(N, model_A, [-12, 0, 0]) #np.random.random(3))
params_B = metropolis(N, model_B, np.random.random(3))
params_C = metropolis(N, model_C, np.random.random(5))

bic_A = -loglike(X, Y, sigma_y, [np.mean(params_A[:,0]), np.mean(params_A[:,1]), np.mean(params_A[:,2])], model_A) + (3/2)*np.log(len(X))
bic_B = -loglike(X, Y, sigma_y, [np.mean(params_B[:,0]), np.mean(params_B[:,1]), np.mean(params_B[:,2])], model_B) + (3/2)*np.log(len(X))
bic_C = -loglike(X, Y, sigma_y,[np.mean(params_C[:,0]),np.mean(params_C[:,1]),np.mean(params_C[:,2]),np.mean(params_C[:,3]),np.mean(params_C[:,4])],model_C)+(5/2)*np.log(len(X))

x_ = np.linspace(3, 7, 100)

plt.figure(figsize=(12,12))
for i in range(3):
    plt.subplot(2,2,i+1)
    plt.hist(params_A[int(N/2):int(N),i],label='Model A',color='orange')
    plt.title(r'$\beta_%1.0f $='%i+'%0.4f' %np.mean(params_A[int(N/2):int(N),i]) + r'$\pm$ %0.4f' %np.std(params_A[int(N/2):int(N),i]))
    plt.xlabel(r'$\beta_%1.0f $'%i)
    plt.legend()
    plt.tight_layout()
plt.subplot(2,2,4)
plt.title('Bic A = %0.1f' %bic_A)
plt.errorbar(X, Y, yerr=sigma_y, xerr=0, fmt='.', label='Data')
plt.plot(x_, model_A(x_, np.mean(params_A[int(N/2):int(N),:],axis=0)),label='Model A')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('modelo_A.png')
plt.show()

plt.figure(figsize=(12,12))
for i in range(3):
    plt.subplot(2,2,i+1)
    plt.hist(params_B[int(N/2):int(N),i],label='Model B',color='blue')
    plt.title(r'$\beta_%1.0f $='%i+'%0.4f' %np.mean(params_B[int(N/2):int(N),i]) + r'$\pm$ %0.4f' %np.std(params_B[int(N/2):int(N),i]))
    plt.xlabel(r'$\beta_%1.0f $'%i)
    plt.legend()
    plt.tight_layout()
plt.subplot(2,2,4)
plt.title('Bic B = %0.1f' %bic_B)
plt.errorbar(X, Y, yerr=sigma_y, xerr=0, fmt='.', label='Data')
plt.plot(x_, model_B(x_, np.mean(params_B[int(N/2):int(N),:],axis=0)),label='Model B')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('modelo_B.png')
plt.show()

plt.figure(figsize=(12,12))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.hist(params_C[int(N/2):int(N),i],label='Model C', color='red')
    plt.title(r'$\beta_%1.0f $='%i+'%0.4f' %np.mean(params_C[int(N/2):int(N),i]) + r'$\pm$ %0.4f' %np.std(params_C[int(N/2):int(N),i]))
    plt.xlabel(r'$\beta_%1.0f $'%i)
    plt.legend()
    plt.tight_layout()
plt.subplot(2,3,6)
plt.title('Bic C = %0.1f' %bic_C)
plt.errorbar(X, Y, yerr=sigma_y, xerr=0, fmt='.', label='Data')
plt.plot(x_, model_C(x_, np.mean(params_C[int(N/2):int(N),:],axis=0)),label='Model C')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('modelo_C.png')
plt.show()