import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import numpy.random

plt.rcParams.update({'font.size': 30})
plt.rcParams["font.family"] = "Times New Roman"

discount = 0.6
from copy import deepcopy
HParray = np.array([[4,10,10]])
#HParray = [(20,20,10),(20,50,10),(10,5,5),(10,20,10),(100,20,20)]
max_iterations = 2000000
SORAverages = 10
seedArray = np.arange(SORAverages)
print(seedArray)
QlearningError = np.zeros((SORAverages, max_iterations))
QlearningErrorSOR = np.zeros((SORAverages, max_iterations))
# print("Exiting")

for seed in seedArray:
    numpy.random.seed(seed)
    horizons = HParray[0, 0]
    No_of_states = HParray[0, 1]
    No_of_actions = HParray[0, 2]
    print(horizons,No_of_states,No_of_actions)
    # max_iterations = 2000000

    states = np.arange(No_of_states)
    actions = np.arange(No_of_actions)


    costVector = np.random.rand(horizons+1,No_of_states,No_of_actions)*10

    wMatrix = np.ones([horizons + 1])
    wOptimal = 1.4
    pOptimal = (1 - (1 / wOptimal)) / discount

    Probbase = np.random.rand(horizons+1,No_of_states,No_of_actions,No_of_states)
    print(Probbase.shape)
    ProbMatrix = np.apply_along_axis(lambda x: x/np.sum(x), 3, Probbase)
    print(ProbMatrix.shape)
    for h in range(horizons+1):
        for s in range(No_of_states):
            for a in range(No_of_actions):
                # difference = ProbMatrix[h,s,a,s] - pOptimal
                probVal = ProbMatrix[h,s,a,s]
                for s1 in range(No_of_states):
                    if(s==s1):
                        ProbMatrix[h,s,a,s1] = pOptimal
                    else:
                        ProbMatrix[h,s,a,s1] = ProbMatrix[h,s,a,s1]*((1-pOptimal)/(1-probVal))
    #ProbMatrix[h, s, a, s1] = ProbMatrix[h, s, a, s1] + difference / (No_of_states - 1)
    #ProbMatrix = np.apply_along_axis(lambda x: x / np.sum(x), 3, Probbase)
    print(ProbMatrix[2,3,2,3])

    for h in range(horizons):
        for s in range(No_of_states):
            for a1 in range(No_of_actions):
                print("test", a1)
                ProbMatrix[h,s,a1] = np.random.random(No_of_states)

                ProbMatrix[h][s][a1][s] = No_of_states  # check this.
                ProbMatrix[h][s][a1] = ProbMatrix[h,s,a1] / ProbMatrix[h,s,a1].sum()

    for s in range(No_of_states):
        for a in range(No_of_actions):
            for s1 in range(No_of_states):
                if(s==s1):
                    ProbMatrix[horizons, s, a, s1] = 1
                else:
                    ProbMatrix[horizons, s, a, s1] = 0

    for h in range(horizons+1):
        for s in range(No_of_states):
            for a in range(No_of_actions):
                print(np.sum(ProbMatrix[h,s,a,:]))
    print(pOptimal)

    print(wMatrix)
    for h in range(0,horizons+1):
        min2 = math.inf
        for s in range(No_of_states):
            for a in range(No_of_actions):
                wVal = 1 / (1 - discount * ProbMatrix[h,s,a,s])
                if wVal < min2:
                    min2 = wVal
        wMatrix[h] = min2

    print(wMatrix)
    print(np.sum(ProbMatrix))
    print((horizons+1)*No_of_states*No_of_actions)

    terminalCost = np.arange(No_of_states)
# terminalCost[0]=1
# terminalCost[1]=2
# terminalCost[2]=3
# terminalCost[3]=4
# terminalCost[4]=5
# terminalCost[5]=4
# terminalCost[6]=3
# terminalCost[7]=2
# terminalCost[8]=1
# terminalCost[9]=1

    QMatrix_current = np.zeros([horizons+1,No_of_states,No_of_actions])
    QMatrix_current_SOR = np.zeros([horizons + 1, No_of_states, No_of_actions])
    # QMatrix_prev = np.ones([horizons+1,No_of_states,No_of_actions])
    QMatrix_DP = np.zeros([horizons+1,No_of_states,No_of_actions])
    #QMatrix_DP_prev = np.ones([horizons+1,No_of_states,No_of_actions])
    tot_count = np.zeros((horizons + 1, No_of_states, No_of_actions, No_of_states))

    for s in range(No_of_states):
        for a in range(No_of_actions):
            QMatrix_current[horizons,s,a]= terminalCost[s]
            #QMatrix_current_SOR[horizons,s,a]= terminalCost[s]
            # QMatrix_prev[horizons,s,a] = terminalCost[s]
            QMatrix_DP[horizons,s,a] = terminalCost[s]
            QMatrix_current_SOR[horizons, s, a] = terminalCost[s]

            # QMatrix_DP_current[horizons, s, a] = terminalCost[s]
            # QMatrix_DP_prev[horizons, s, a] = terminalCost[s]

    def sampleState(h, s, a):
        return np.random.choice(states, 1, p=ProbMatrix[h,s,a,:])

    def stepSize(n):
        return 1/math.ceil((n+1)/10)
    #print(np.sum(QMatrix_DP_prev))
    #print(np.sum(QMatrix_DP_current))
    print("Value Iteration")

#Dynamic Programming

    for h in reversed(range(horizons)):
        for s in range(No_of_states):
            for a in range(No_of_actions):
                for snext in range(No_of_states):
                    QMatrix_DP[h, s, a] += ProbMatrix[h, s, a, snext]*(costVector[h, s, a] + discount*np.amin(QMatrix_DP[h+1, snext, :]))
                    #print(np.sum(QMatrix_DP_current), np.sum(QMatrix_DP_prev))
                    #print(np.linalg.norm(QMatrix_DP_current - QMatrix_DP_prev))

        # if np.linalg.norm(QMatrix_DP_current-QMatrix_DP_prev) < 0.0001:
        #     break

    policyMatrixDP = np.zeros([(horizons + 1), No_of_states])
    valueMatrixDP = np.zeros([horizons + 1, No_of_states])
    for h in range(horizons+1):
        for s in range(No_of_states):
            policyMatrixDP[h, s] = np.argmin(QMatrix_DP[h, s, :])
            valueMatrixDP[h, s] = np.amin(QMatrix_DP[h, s, :])

    # QlearningError = np.zeros(max_iterations)
    # QlearningErrorSOR = np.zeros(max_iterations)

    policyMatrixLearned = np.zeros([horizons + 1, No_of_states])
    valueMatrixLearned = np.zeros([horizons + 1, No_of_states])
    policyMatrixSOR = np.zeros([horizons+1, No_of_states])
    valueMatrixSOR = np.zeros([horizons+1, No_of_states])


#Q-learning
    h = 0
    state = np.random.randint(0, No_of_states)
    for m in range(max_iterations):
        if m % 10000 == 0:
            print("Iteration ", m, flush=True)
        # print(n)
        # be careful with h=horizon case
        if (h >= horizons):
            h = np.random.randint(0, horizons)
            state = np.random.randint(0, No_of_states)

        a = np.random.randint(0, No_of_actions)  # numpy has same function , don't confuse
        s_new = int(np.random.choice(np.arange(No_of_states), 1, p=ProbMatrix[h, state, a, :]))


        #r = R[h][state][act1][act2]
        r = costVector[h,state,a]

        # print(Q[s_new,:,:])

        tot_count[h][state][a][s_new] += 1

        next_state_value = np.amin(QMatrix_current[h + 1, s_new, :])

        next_state_value_SOR = np.amin(QMatrix_current_SOR[h + 1, s_new, :])
        current_state_value = np.amin(QMatrix_current_SOR[h, state, :])
        print(current_state_value)
        w = 1.4

            # Q update
            # print(np.sum(tot_count[h][state][act1][act2]))
        step = stepSize(np.sum(tot_count[h][state][a])) #This works and gives sum over all snew when for [h][state][a]
        d = w*(r + discount*next_state_value_SOR) + (1-w)* current_state_value #mistake

        QMatrix_current_SOR[h, state, a] = (1 - step) * QMatrix_current_SOR[h, state, a] + step * d

        QMatrix_current[h, state, a] = (1 - step) * QMatrix_current[h, state, a] + step * (r + discount*next_state_value)

        policyMatrixLearned[h, state] = np.argmin(QMatrix_current[h, state, :])
        valueMatrixLearned[h, state] = np.amin(QMatrix_current[h, state, :])
        policyMatrixSOR[h, state] = np.argmin(QMatrix_current_SOR[h, state, :])
        valueMatrixSOR[h, state] = np.amin(QMatrix_current_SOR[h, state, :])

        error1 = np.linalg.norm(valueMatrixLearned - valueMatrixDP)
        error2 = np.linalg.norm(valueMatrixSOR - valueMatrixDP)

        QlearningError[seed,m] = error1/(math.sqrt((horizons+1)*No_of_states))
        QlearningErrorSOR[seed,m] = error2/(math.sqrt((horizons+1)*No_of_states))

            # print("hihi Q")
            # print(np.sum(Q))

        h = h + 1
        state = s_new

    np.save('FHQLError_{}_{}_{}.npy'.format(horizons,No_of_states,No_of_actions),QlearningError)

    #
    # for h in range(horizons+1):
    #     for s in range(No_of_states):
    #         policyMatrixLearned[h, s] = np.argmin(QMatrix_current[h, s, :])
    #         valueMatrixLearned[h, s] = np.amin(QMatrix_current[h, s, :])
    #         policyMatrixSOR[h, s] = np.argmin(QMatrix_current_SOR[h, s, :])
    #         valueMatrixSOR[h, s] = np.amin(QMatrix_current_SOR[h, s, :])

            # aOptimal = policyMatrixLearned[h,s]
            # valueMatrixSOR[h,s] = QM

#Plot error
    print("hihi")
    print(np.sum(valueMatrixSOR))

    print(valueMatrixSOR[1,:])
    print(valueMatrixLearned[1,:])
    print(valueMatrixDP[1,:])
    #print(valueMatrixSOR[1,:])
    print(valueMatrixSOR[1,:]-valueMatrixDP[1,:])
    print(valueMatrixLearned[1,:]- valueMatrixDP[1,:] )
    print()

    midHorizon = math.ceil(horizons/2)
# #Plotting value Function
#     plot1 = plt.figure()
#     plt.xticks(np.arange(0, stop = No_of_states+1 , step = 2))
#     #plt.xticks(rotation=90)
#     plt.plot(np.arange(No_of_states), valueMatrixSOR[0,:], '*',label='SOR ')
#     plt.plot(np.arange(No_of_states), valueMatrixLearned[0, :], '-',label='Learned ')
#     plt.plot(np.arange(No_of_states), valueMatrixDP[0, :],'--',label='DP')
# #plt.plot([],[],label='horizons = {}, states = {}, actions = {}'.format(horizons,No_of_states,No_of_actions))
#
#
#     plt.title('Optimal value function $J_0$: N = {}, |S| = {}, |A| = {}'.format(horizons, No_of_states, No_of_actions))
#     plt.xlabel('States')
#     plt.ylabel('Optimal Value ')
#     #plt.figlegend()
#     leg = plt.legend()
#     leg.set_draggable(state=True)
#     #plt.savefig('value_{}_{}_{}_async'.format(horizons,No_of_states,No_of_actions))
# #plt.show()
#
#     plot2 = plt.figure()
#     plt.plot(np.arange(No_of_states), policyMatrixSOR[0, :], '*', label='SOR ')
#     plt.plot(np.arange(No_of_states), policyMatrixLearned[0, :], '-',label='Learned ')
#     plt.plot(np.arange(No_of_states), policyMatrixDP[0, :],'--',label='DP')
#     plt.xticks(np.arange(No_of_states))
#     plt.yticks(np.arange(0, stop=No_of_actions+1, step =2))
#     plt.title('Optimal Policy $\pi_0$: N = {}, |S| = {}, |A| = {}'.format(horizons, No_of_states, No_of_actions))
#     plt.xlabel('States')
#     plt.ylabel('Optimal Action ')
#     #plt.figlegend()
#     leg = plt.legend(fontsize=28)
#     leg.set_draggable(state=True)
#     #plt.savefig('policy_{}_{}_{}_async'.format(horizons,No_of_states,No_of_actions))

plot3 = plt.figure()
#plt.axis([0, min(5000,max_iterations), 0, 0.50])
x = np.arange(max_iterations)
print(x.shape)
print(QlearningError.shape)

#plt.axis([0, m, 0, 50])
plt.plot(np.average(QlearningError, axis=0), label= 'Finite horizon Q-learning')
plt.yscale("log")
# plot6 = plt.figure()
# x = np.arange(max_iterations)
plt.plot(np.average(QlearningErrorSOR, axis=0) , label='Finite horizon SOR Q-learning')
plt.yscale("log")
#plt.xticks(list(range(1,max(x)+1)), [str(i) for i in range(1,max(x)+1)])
plt.title("Error for N = {}, |S| = {}, |A|= {} ".format(horizons, No_of_states, No_of_actions))
plt.xlabel("Number of iterations")
plt.ylabel("Error")
#plt.figlegend()
leg = plt.legend()
leg.set_draggable(state=True)
#plt.savefig('error_{}_{}_{}_async'.format(horizons,No_of_states,No_of_actions))
#plt.xlim([1,20000])
plt.show()

print(m)
print(np.linalg.norm(QMatrix_current-QMatrix_DP))

print(np.linalg.norm(QMatrix_current[0]-QMatrix_DP[0]))
print(np.sum(QMatrix_DP))
print(np.sum(QMatrix_current))
print(np.sum(QMatrix_current_SOR))
print(np.sum(valueMatrixLearned))
print(np.sum(valueMatrixSOR))
print(np.sum(valueMatrixDP))
print("Done")