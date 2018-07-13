import numpy as np
import pickle


num_tasks = 10
num_objects = 4

goals = []

for t in range(num_tasks):
    goal_pos = np.zeros(2)
    object_pos = np.zeros(num_objects * 2)
    for l in range(0,num_objects * 2,2):
        keep_sampling = True
        while keep_sampling:
            i = np.random.randint(3)
            if l == 0:
                i == 1

            print('i', i)
            print('l', l)

            if i == 0  and l is not 0:

                object_pos[l] = np.random.uniform(0.15, 0.3)
                object_pos[l + 1] = np.random.uniform(0.6, 0.8)

            elif i == 1 and l == 0:

                object_pos[l] = np.random.uniform(-0.15, 0.15)
                object_pos[l + 1] = np.random.uniform(0.65, 0.75)


            elif i == 2 and l is not 0:

                object_pos[l] = np.random.uniform(-0.3, -0.15)
                object_pos[l + 1] = np.random.uniform(0.6, 0.8)


            print(object_pos[l])
            #import IPython
            #IPython.embed()

            if np.linalg.norm(object_pos[l:l+2]) > 0:
                #import IPython
                #IPython.embed()
                keep_sampling = False

                if l == 0:
                    goal_pos[0] = object_pos[l] + np.random.uniform(-0.1, 0.1)
                    goal_pos[1] = object_pos[l+1] +  0.2 + np.random.uniform(-0.1, 0.1)

                if l > 0 and np.linalg.norm(object_pos[0:2] - object_pos[l: l+2]) > 0.3:
                    print(np.linalg.norm(object_pos[0:1] - object_pos[l: l+1]))
                    keep_sampling = False



    goals.append(np.concatenate(( goal_pos, object_pos), axis=0))

    print(goals)


import matplotlib.pyplot as plt
for l in range(num_tasks):

    #import IPython
    #IPython.embed()
    plt.scatter(np.array(goals[l])[2], np.array(goals[l])[3], marker='s', c='g')
    plt.scatter(np.array(goals[l])[4:-1:2], np.array(goals[l])[5::2], marker='s')
    plt.scatter(np.array(goals[l])[0], np.array(goals[l])[1], marker='o')

    #plt.show()

pickle.dump(np.asarray(goals), open("goals_sawyer_pusher.pkl", "wb"))
