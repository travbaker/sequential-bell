#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:19:18 2021


Numerics for the second case which maximize the tradeoffs between Eq 11 and 12 in the preprint.
All observables searched over.
Search space is just A1s and B1s measurements. 4 params *4 = 16 total, plus 1 for state.
Search vector is x, blocks of 4. [Bias, strength, two for unit vector]

@author: travis.baker@griffithuni.edu.au
"""
#%%
import numpy as np
import qutip as qt
from scipy.optimize import differential_evolution 
import time
from scipy.linalg import norm
from scipy.optimize import NonlinearConstraint
import sys

Id, sx, sy, sz = qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()

def generic_entangled(alpha):
    dm = np.cos(alpha)*qt.tensor(qt.basis(2,0),qt.basis(2,0)) + np.sin(alpha)*qt.tensor(qt.basis(2,1),qt.basis(2,1))
    return qt.ket2dm(dm)

def ang_to_vec(theta, phi):
    v = np.array([[np.sin(theta)*np.cos(phi)], \
                  [np.sin(theta)*np.sin(phi)], \
                      [np.cos(theta)]])
    return v

def ang_to_mat(theta, phi):
    v = np.array([[np.sin(theta)*np.cos(phi)], \
                  [np.sin(theta)*np.sin(phi)], \
                      [np.cos(theta)]])
    return v@v.T


def reversibility(B,S):
    x = np.max([(1+B)**2 - S**2, 0])
    y = np.max([(1-B)**2 - S**2,0])
    return 1/2*np.sqrt(x) + 1/2*np.sqrt(y)

#Eq 11
def proxy_12(x):
    R_y1, R_y2 = reversibility(x[8],x[9]), reversibility(x[12],x[13])
    T = np.diag([2*np.sin(x[-1])*np.cos(x[-1]), -2*np.sin(x[-1])*np.cos(x[-1]), 1])
    L = 1/2*(R_y1 + R_y2)*np.eye(3) + 1/2*(1 - R_y1)*ang_to_mat(x[10], x[11]) + 1/2*(1 - R_y2)*ang_to_mat(x[14], x[15])
    bz = np.cos(2*x[-1])
    b = np.array([[0], [0], [bz]])
    proxy = norm((x[0] + x[4])*L@b + L@(T.T)@(x[1]*ang_to_vec(x[2],x[3]) + x[5]*ang_to_vec(x[6],x[7]))) + \
            norm((x[0] - x[4])*L@b + L@(T.T)@(x[1]*ang_to_vec(x[2],x[3]) - x[5]*ang_to_vec(x[6],x[7])))
    return np.array(proxy)

#Eq 12
def proxy_21(x, sign=-1):
    R_x1, R_x2 = reversibility(x[0],x[1]), reversibility(x[4],x[5])
    T = np.diag([2*np.sin(x[-1])*np.cos(x[-1]), -2*np.sin(x[-1])*np.cos(x[-1]), 1])
    K = 1/2*(R_x1 + R_x2)*np.eye(3) + 1/2*(1 - R_x1)*ang_to_mat(x[2], x[3]) + 1/2*(1 - R_x2)*ang_to_mat(x[6], x[7])
    az = np.cos(2*x[-1])
    a = np.array([[0], [0], [az]])
    proxy = norm((x[8] + x[12])*K@a + K@T@(x[9]*ang_to_vec(x[10],x[11]) + x[13]*ang_to_vec(x[14],x[15]))) + \
            norm((x[8] - x[12])*K@a + K@T@(x[9]*ang_to_vec(x[10],x[11]) - x[13]*ang_to_vec(x[14],x[15])))
    return sign*proxy

def bias_constraint_1(x):
    return np.array(np.abs(x[0]) + x[1])

def bias_constraint_2(x):
    return np.array(np.abs(x[4]) + x[5])

def bias_constraint_3(x):
    return np.array(np.abs(x[8]) + x[9])

def bias_constraint_4(x):
    return np.array(np.abs(x[12]) + x[13])

def alpha_constraint(x):
    return np.array(x[-1])

proxy_12_min = float(sys.argv[1])
CHSH_nlc = NonlinearConstraint(proxy_12, proxy_12_min, np.inf)
nlc1 = NonlinearConstraint(bias_constraint_1, 0, 1)
nlc2 = NonlinearConstraint(bias_constraint_2, 0, 1)
nlc3 = NonlinearConstraint(bias_constraint_3, 0, 1)
nlc4 = NonlinearConstraint(bias_constraint_4, 0, 1)

bounds=[(-1,1),(0,1),(0,np.pi),(0,2*np.pi)]*4 + [(0,np.pi/2)]# + [(0,np.pi)]*3

def printCurrentIteration(xk, convergence):
    # print('finished iteration')
    # print(xk)
    print('congervence: %.4f' %convergence)

def uniform_init_sampling(num_population_members):
    rng = np.random.default_rng()
    N = 4*(num_population_members-1)
    biases = 1-2*rng.uniform(size=N)
    strengths = 1 - np.abs(biases)*rng.uniform(size=N)
    thetas = np.pi*rng.uniform(size=N)
    phis = 2*np.pi*rng.uniform(size=N)
    alphas = rng.uniform(size=num_population_members-1)
    arr = np.vstack([biases, strengths, thetas, phis])
    arr = np.reshape(arr, (int(N/4), 16))
    arr = np.hstack([arr, np.vstack(alphas)])
    first_member = np.array([0,1,0,0,0,1,np.pi/2,0,0,1,np.pi/4,0,0,1,-np.pi/4,0,0.5])
    inits = np.vstack([first_member, arr])
    return inits

DE_tol=1e-8
POP=400 #20*(2*ma)
start = np.array([0,1,0,0,0,1,np.pi/2,0,0,0,np.pi/4,0,0,0,-np.pi/4,0,np.pi/4])
inits = np.vstack([start, uniform_init_sampling(POP-1)])
start = time.time()
if __name__ == "__main__":
    root=differential_evolution(proxy_21, bounds, args=(), 
                                strategy='best1bin', 
                                maxiter=100000, 
                                popsize=POP, 
                                tol=DE_tol, 
                                mutation=(0.5,1), 
                                recombination=0.7, seed=None, 
                                callback=printCurrentIteration, 
                                disp=True, polish=False, init=inits, atol=0, updating='deferred', workers=-1, 
                                constraints=(CHSH_nlc, nlc1, nlc2, nlc3, nlc4))
    print('S*_21 value: %.4f' %(-root.fun))
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
root_polished = minimize(proxy_21, root.x, args=(), method='SLSQP', jac=None, bounds=bounds, 
                         constraints=({'type': 'ineq', 'fun': lambda x:  1 - np.abs(x[0]) - x[1]},
                              {'type': 'ineq', 'fun': lambda x:  np.abs(x[0]) + x[1]}, 
                              {'type': 'ineq', 'fun': lambda x:  1 - np.abs(x[4]) - x[5]},
                              {'type': 'ineq', 'fun': lambda x:  np.abs(x[4]) + x[5]},
                              {'type': 'ineq', 'fun': lambda x:  1 - np.abs(x[8]) - x[9]},
                              {'type': 'ineq', 'fun': lambda x:  np.abs(x[8]) + x[9]}, 
                              {'type': 'ineq', 'fun': lambda x:  1 - np.abs(x[12]) - x[13]},
                              {'type': 'ineq', 'fun': lambda x:  np.abs(x[12]) + x[13]},
                              {'type': 'ineq', 'fun': lambda x:  proxy_12(x) - proxy_12_min}), 
                         tol=1e-5, callback=None, 
                         options={'maxiter': 1000, 'ftol': 1e-08, 'iprint': 1, 'disp': True, 'eps': 1e-08, 'finite_diff_rel_step': None})

print('CHSH value from SLSQP: %.4f' %(-root_polished.fun))
print('%s \n' %root_polished)
end = time.time()

if root.success==True and root_polished.success==True:
    print('Both converged \n')
    outputs = open('converged_data_disordered.txt', 'a')
    outputs.write('%.8f, %.8f \n' %(proxy_12_min, -root_polished.fun))
    outputs.close()
elif root.success==True:
    print('Only DE converged \n')
    outputs = open('converged_data_disordered.txt', 'a')
    outputs.write('%.8f, %.8f \n' %(proxy_12_min, -root.fun))
    outputs.close()
else:
    print('None converged \n')
    outputs = open('converged_data_disordered.txt', 'a')
    outputs.write('Non-convergent at %.8f \n' %(proxy_12_min))
    outputs.close()

print('SLSQP time: %2f minutes' %((end - start)/60))


