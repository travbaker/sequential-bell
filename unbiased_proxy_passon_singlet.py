#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 02:45:00 2021

Maximize the tradeoff between S*(A2,B2) and S(A1,B1) when A1,B1 restricted to measure unbiased observables, on a singlet state.

@author: travis.baker@griffithuni.edu.au
"""
#%%
import numpy as np
import qutip as qt
from scipy.optimize import differential_evolution 
import time
import sys
from scipy.linalg import svdvals 
from scipy.optimize import NonlinearConstraint

Id, sx, sy, sz = qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()

def generic_entangled(alpha):
    dm = np.cos(alpha)*qt.tensor(qt.basis(2,0),qt.basis(2,0)) + np.sin(alpha)*qt.tensor(qt.basis(2,1),qt.basis(2,1))
    return qt.ket2dm(dm)

def CHSH_first_eval(x):
    X1 = x[0]*(np.sin(x[1])*np.cos(x[2])*sx + np.sin(x[1])*np.sin(x[2])*sy + np.cos(x[1])*sz)
    X2 = x[3]*(np.sin(x[4])*np.cos(x[5])*sx + np.sin(x[4])*np.sin(x[5])*sy + np.cos(x[4])*sz)
    Y1 = x[6]*(np.sin(x[7])*np.cos(x[8])*sx + np.sin(x[7])*np.sin(x[8])*sy + np.cos(x[7])*sz)
    Y2 = x[9]*(np.sin(x[10])*np.cos(x[11])*sx + np.sin(x[10])*np.sin(x[11])*sy + np.cos(x[10])*sz)
    bell_op = qt.tensor(X1, Y1) + qt.tensor(X1, Y2) + qt.tensor(X2, Y1) - qt.tensor(X2, Y2)
    rho_initial = generic_entangled(np.pi/4)
    val = np.real(np.trace(rho_initial*bell_op))
    return np.abs(np.array(val))

S_1_min = float(sys.argv[1])
CHSH_nlc = NonlinearConstraint(CHSH_first_eval, S_1_min, np.inf)

def reversibility(B,S):
    x = np.max([(1+B)**2 - S**2, 0])
    y = np.max([(1-B)**2 - S**2,0])
    return 1/2*np.sqrt(x) + 1/2*np.sqrt(y)

def ang_to_vec(theta, phi):
    v = np.array([[np.sin(theta)*np.cos(phi)], \
                  [np.sin(theta)*np.sin(phi)], \
                      [np.cos(theta)]])
    return v@v.T

#A2 and B2's measurements are projective. so only 4 params total.
def eq10(x,sign=-1): 
    R_x1, R_x2 = reversibility(0,x[0]), reversibility(0,x[3])
    R_y1, R_y2 = reversibility(0,x[6]), reversibility(0,x[9])
    T = np.diag([1, -1, 1])
    K = 1/2*(R_x1 + R_x2)*np.eye(3) + 1/2*(1 - R_x1)*ang_to_vec(x[1], x[2]) + 1/2*(1 - R_x2)*ang_to_vec(x[4], x[5])
    L = 1/2*(R_y1 + R_y2)*np.eye(3) + 1/2*(1 - R_y1)*ang_to_vec(x[7], x[8]) + 1/2*(1 - R_y2)*ang_to_vec(x[10], x[11])
    vals = svdvals(K@T@L)
    obj = 2*np.sqrt(vals[0]**2 + vals[1]**2)
    # print(obj)
    return sign*obj

bnds=[(0,1),(0,np.pi),(0,2*np.pi)]*4

def printCurrentIteration(xk, convergence):
    # print('finished iteration')
    # print(xk)
    print('congervence: %.8f' %convergence)

def uniform_init_sampling(num_population_members):
    rng = np.random.default_rng()
    N = 4*(num_population_members-1)
    # biases = 1-2*rng.uniform(size=N)
    strengths = rng.uniform(size=N)
    thetas = np.pi*rng.uniform(size=N)
    phis = 2*np.pi*rng.uniform(size=N)
    # alphas = rng.uniform(size=num_population_members-1)
    arr = np.vstack([strengths, thetas, phis])
    arr = np.reshape(arr, (int(N/4), 12))
    # arr = np.hstack([arr, np.vstack(alphas)])
    first_member = np.array([1,0,0,1,np.pi/2,0,1,np.pi/4,0,1,-np.pi/4,0])
    inits = np.vstack([first_member, arr])
    return inits

DE_tol=1e-5
mcPOP=0 #20*(2*ma)
randPOP=200
POP=mcPOP+randPOP    

# sample=np.loadtxt('r/sample_matrix_%.8f.txt' %S_1_min)
# init_sample = np.empty((POP,17))
# for i in range(mcPOP):
#     init_sample[i] = np.insert(sample[i], [0,3,6,9], [0])

#inits = init_sample
# inits = np.vstack([init_sample, uniform_init_sampling(randPOP)])
inits = uniform_init_sampling(randPOP)
start = time.time()
if __name__ == "__main__":
    root=differential_evolution(eq10, bnds, args=(), 
                                strategy='best1bin', 
                                maxiter=100000, 
                                popsize=POP, 
                                tol=DE_tol, 
                                mutation=(0.5,1), 
                                recombination=0.7, seed=None, 
                                callback=printCurrentIteration, 
                                disp=True, polish=False, init=inits, atol=0, updating='deferred', workers=-1, 
                                constraints=(CHSH_nlc))
    print('CHSH value from DE: %.4f' %(-root.fun))
end = time.time()

print('%s \n' %root)
print('DE time: %2f minutes' %((end - start)/60))
print('DE tol is %.8f' %DE_tol)
print('DE pop is %d' %POP)


# =============================================================================
# Polish
# =============================================================================

from scipy.optimize import minimize
start = time.time()
root_polished = minimize(eq10, root.x, args=(), method='SLSQP', jac=None, bounds=bnds, 
                         constraints=({'type': 'ineq', 'fun': lambda x:  CHSH_first_eval(x) - S_1_min}), 
                         tol=1e-8, callback=None, 
                         options={'maxiter': 1000, 'ftol': 1e-08, 'iprint': 1, 'disp': True, 'eps': 1e-08, 'finite_diff_rel_step': None})
print(root_polished)
print('CHSH value from SLSQP: %.4f' %(-root_polished.fun))
end = time.time()

#%%
if root.success==True and root_polished.success==True:
    print('Both converged \n')
    outputs = open('unbiased_singlet_converged_data.txt', 'a') 
    outputs.write('%.8f, %.8f \n' %(S_1_min, -root_polished.fun)) 
    outputs.close()
elif root.success==True:
    print('Only DE converged \n')
    outputs = open('unbiased_singlet_converged_data.txt', 'a')
    outputs.write('%.8f, %.8f \n' %(S_1_min, -root.fun))
    outputs.close()
else:
    print('None converged \n')
    outputs = open('unbiased_singlet_converged_data.txt', 'a')
    outputs.write('Non-convergent at %.8f \n' %(S_1_min))
    outputs.close()

print('SLSQP time: %2f minutes' %((end - start)/60))
