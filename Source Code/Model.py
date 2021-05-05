# Daniel Roberts
# Daoqun Yang
# Deep Q-Learning

import numpy as np
import gym
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Reshape
from keras.models import Sequential


class ModelBuild:
    def __init__(self, maxSize, numAct):
        self.memSize = maxSize
        self.numAct = numAct
        shape = 4

        # Starting with all zeros for random action
        self.currSpace = np.zeros((self.memSize, shape))
        self.newSpace = np.zeros((self.memSize, shape))
        self.actionMem = np.zeros(self.memSize)
        self.rewardMem = np.zeros(self.memSize)
        self.endMem = np.zeros(self.memSize)
        self.count = np.zeros(2, dtype = "int32")

    def storeMem(self, curr, action, reward, new, end):

        # Checking to see if indexing is needed
        if (self.count[0] == self.memSize):
            self.count[0] = 0
            self.count[1] += 1

        # Adding item or overwriting
        self.actionMem[self.count[0]] = action
        self.currSpace[self.count[0]] = curr
        self.rewardMem[self.count[0]] = reward
        self.newSpace[self.count[0]] = new
        self.endMem[self.count[0]] = 1 - int(end)

        # Incrimenting
        self.count[0] += 1

    def sample(self, size):

        # Generating variables
        curr = []
        new = []
        reward = []
        action = []
        end = []

        # Getting random indicies
        index = np.random.choice(range(self.count[0]), size = size)

        # Getting sample
        for i in index:
            curr.append(self.currSpace[i])
            new.append(self.newSpace[i])
            reward.append(self.rewardMem[i])
            action.append(self.actionMem[i])
            end.append(self.endMem[i])

        return curr, new, reward, action, end

    def dqnModel(self):

        # Model with two hidden layers using relu and input, reshaping, and output layers
        model = Sequential([
            Embedding(500, 10, input_length = 1),
            Reshape((10, )),
            Dense(55, activation = "linear"),
            Dense(95, activation = "relu"),
            Dense(self.numAct)])

        model.compile(loss = "mse")

        return model

class Learner:
    def __init__(self, maxSize, numAct, gamma, epsilon, sampleSize, epsilonDec, epsilonEnd):
        self.gamma = gamma
        self.epsilon = epsilon
        self.sampleSize = sampleSize
        self.epsilonDec = epsilonDec
        self.epsilonEnd = epsilonEnd
        self.numAct = numAct

        self.build = ModelBuild(maxSize, numAct)

        self.model = self.build.dqnModel()
    
    def choose(self, state):

        # Picking random number
        rand = np.random.random()

        # Checking rand against epsilon
        if (rand < self.epsilon):
            action = np.random.choice(self.numAct)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions[0])

        return action

    def learing(self):

        # Making sure we have enough data to sample
        if (self.build.count[0] % self.sampleSize == 0):
            # Sampling
            currs, news, rewards, actions, ends = self.build.sample(self.sampleSize)

            # Looping though sample
            for curr, new, reward, action, end in zip(currs, news, rewards, actions, ends):
                # Getting target model
                target = self.model.predict(curr)

                # Predicting based on new state
                nextPred = self.model.predict(new)

                # Q equation
                target[0][int(action)] = reward + self.gamma * np.max(nextPred) * end
            
            # Fitting new model
            self.model.fit(curr, target, verbose = 0)

            # Adjusting epsilon value
            if (self.epsilon > self.epsilonEnd):
                self.epsilon = self.epsilon * self.epsilonDec
            else:
                self.epsilon = self.epsilonEnd

    def rewardCheck(self, new, curr, action, reward):

        # Checking if dropped off or picked up
        if (action == 4 or action == 5):
            return reward

        # Added negative reward for repeat actions to encourage more exploration
        elif (self.build.count[0] >= 3 and action == self.build.actionMem[self.build.count[0]] and 
            self.build.actionMem[self.build.count[0]] == self.build.actionMem[self.build.count[0] - 1] and
            self.build.actionMem[self.build.count[0] - 1] == self.build.actionMem[self.build.count[0] - 2] and
            self.build.actionMem[self.build.count[0] - 2] == self.build.actionMem[self.build.count[0] - 3]):

            return -6

        # Gives points on a successful pickup
        elif (curr[2] != 4 and new[2] == 4 and action == 4):
            return 10
        
        # Giving neutral reward for being within one space of the pickup if picking up and one space of the delivery if passenger is on board
        elif (new[2] != 4):
            
            # When pickup is at top left
            if (new[2] == 0):

                # When taxi is in first or second row and first or second column
                if ((new[0] == 0 or new[0] == 1) and (new[1] == 0 or new[1] == 1)):
                    return 1
                else:
                    return reward

            # When pickup is top right
            elif (new[2] == 1):
                if ((new[0] == 0 or new[0] == 1) and (new[1] == 3 or new[1] == 4)):
                    return .5
                else:
                    return reward

            # When pickup is bottom left
            elif (new[2] == 2):
                if ((new[0] == 3 or new[0] == 4) and (new[1] == 0 or new[1] == 1)):
                    return .5
                else:
                    return reward

            # When pickup is bottom right
            elif (new[2] == 3):
                if ((new[0] == 3 or new[0] == 4) and (new[1] == 3 or new[1] == 4)):
                    return .5
                else:
                    return reward

        elif (new[2] == 4):

            # When dropoff is at top left
            if (new[3] == 0):
                if ((new[0] == 0 or new[0] == 1) and (new[1] == 0 or new[1] == 1)):
                    return .5
                else:
                    return reward

            # When dropoff is top right
            elif (new[3] == 1):
                if ((new[0] == 0 or new[0] == 1) and (new[1] == 3 or new[1] == 4)):
                    return .5
                else:
                    return reward

            # When dropoff is bottom left
            elif (new[3] == 2):
                if ((new[0] == 3 or new[0] == 4) and new[1] == 0):
                    return .5
                else:
                    return reward

            # When dropoff is bottom right
            elif (new[3] == 3):
                if ((new[0] == 3 or new[0] == 4) and (new[1] == 3 or new[1] == 4)):
                    return .5
                else:
                    return reward

        else:
            return reward


# Making main function
if __name__ == "__main__":

    modelName = input("Please input model name:")
    fileName = input("Please input file name:")

    # Initializing enviornment
    env = gym.make("Taxi-v3")

    # Setting number of games and intializing class
    numGames = 200

    model = Learner(100000, 6, 0.9, 1, 50, .99, .05)

    scoreHist = ["Score,", "Steps,", "Highscore,", "Life\n"]
    

    # Running simulations
    for i in range(numGames):
        score = 0
        highScore = 0
        end = False
        env.reset()
        curr = env.step(np.random.choice(range(4)))
        curr = list(env.decode(curr[0]))
        start = model.build.count[0]

        # Running until death
        while not end:
            # Getting model recomendation
            action = model.choose(curr)

            # Running model based on recomendation
            new, reward, end, info = env.step(action)

            # Printing iteration and enviornment
            print(model.build.count[0])
            env.render()

            # Decoding new state to values
            new = list(env.decode(new))

            # Editting reward to better help the model learn and avoid sparse rewards
            reward = model.rewardCheck(new, curr, action, reward)

            # Adding on score and storing frame
            score += reward

            # High score
            if (score >= highScore):
                highScore = score

            # Killing it after score is too negative
            if (score <= -500 or score >= 500):
                end = True
            
            else:
                end = False
            
            model.build.storeMem(curr, action, reward, new, end)

            # Setting current to new state
            curr = new

            # Making model learn
            model.learing()
        
        # Appending final game score
        scoreHist.append(str(score) + ",")
        scoreHist.append(str(model.build.count[0] - start) + ",")
        scoreHist.append(str(highScore) + ",")
        scoreHist.append(str(i) + "\n")

        # End condition for winning
        if (highScore == 500):
            break
        
    # Print data of scores
    with open(fileName, "w") as txt:
        txt.writelines(scoreHist)

    # Saving model
    model.model.save(modelName)