import numpy as np
from matplotlib import pyplot as plt


# define prior and signals
S = np.matrix([[1,0],[0,12]]) #prior variance
c1 = [3,1] #source coefficients 1
c2 = [0,1] #source coefficients 2
c3 = [1,0] #source coefficients 3
C = np.matrix([c1,c2,c3]) #3x2 matrix of source coefficient vectors


# define frequency matrices
q_trap = list()
for t in range(1,101):
    q_trap.append(np.matrix([[0,0,0],[0,0,0],[0,0,t]])) #path: x3 -> x3 -> x3 -> x3 -> ...

q_effi = [np.matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])] #initial signals: x1, then x2
for t in range(2,100):
    if (t % 2) == 0:
        q_effi.append(np.add(q_effi[t-1], np.matrix([[1,0,0],[0,0,0],[0,0,0]]))) #add x1 signal
    else:
        q_effi.append(np.add(q_effi[t-1], np.matrix([[0,0,0],[0,1,0],[0,0,0]]))) #add x2 signal
#path: x1 -> x2 -> x1 -> x2 -> x1 -> ...

q_trap_deviation = [np.matrix([[0,0,0],[0,0,0],[0,0,1]]), np.matrix([[1,0,0],[0,0,0],[0,0,1]])] #initial signals: x3, x1
for t in range(2,101):
    if (t % 2) == 0:
        q_trap_deviation.append(np.add(q_trap_deviation[t-1], np.matrix([[0,0,0],[0,1,0],[0,0,0]]))) #add x2 signal
    else:
        q_trap_deviation.append(np.add(q_trap_deviation[t-1], np.matrix([[1,0,0],[0,0,0],[0,0,0]]))) #add x1 signal
# #deviation to "efficient" path after t=1


# compute posterior variances for learning trap and "efficient" learning path
V_trap = [1] #first element: prior variance
for t in range(0,100):
    V_trap.append(np.linalg.inv(np.add(np.linalg.inv(S), np.matmul(np.matmul(np.transpose(C), q_trap[t]), C)))[0,0])

V_effi = [1] #first element: prior variance
for t in range(0,100):
    V_effi.append(np.linalg.inv(np.add(np.linalg.inv(S), np.matmul(np.matmul(np.transpose(C), q_effi[t]), C)))[0,0])

V_trap_deviation = [1] #first element: prior variance, second element: posterior after X3
for t in range(0,100):
    V_trap_deviation.append(np.linalg.inv(np.add(np.linalg.inv(S), np.matmul(np.matmul(np.transpose(C), q_trap_deviation[t]), C)))[0,0])


# plot posterior variance as fct. of learning paths
plt.plot(V_trap_deviation[0:16], '--', color='tab:red') #plot deviation from trap
plt.plot(V_trap[0:16], color='tab:red', marker='o', mfc='tab:red') #plot trap
plt.plot(V_effi[0:16], color='tab:blue', marker='o', mfc='tab:blue') #plot efficient learning
plt.xticks(range(0,16)) #set the tick frequency on x-axis

plt.ylabel('posterior variance of \u03C9', fontsize=13) #set the label for y axis
plt.xlabel('period t', fontsize=13) #set the label for x-axis
plt.title('') #set the title of the graph
plt.grid(axis='both', alpha=.2) #add semi-transparent grid
plt.ylim(0) #truncation of y-axis at 0

plt.show() #display the graph