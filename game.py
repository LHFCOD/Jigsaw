import random
class Squre:
    def __init__(self):
        self.index=0
        self.isSpace=False
class Game:
    Actions=4
    Up=0
    Down=1
    Left=2
    Right=3
    def __init__(self,_width,_height):
        self.width=_width
        self.height=_height
        self.count=(self.width)*(self.height)
        self.SqureContainer={}
        self.spacePos=0
        for i in range(self.height):
            for j in range(self.width):
                squre = Squre()
                squre.index = i + j * self.width
                if i!=0 and j!=0:
                    squre.isSpace=False
                else:
                    squre.isSpace=True
                self.SqureContainer[squre.index] = squre
    def GetSqureFromPos(self,_pos):
        return self.SqureContainer[_pos]
    def GetSqureFromXY(self,_x,_y):
        pos=_x+_y*self.width
        return self.SqureContainer[pos]

    def SetSqureFromPos(self, _pos,_squre):
        self.SqureContainer[_pos]=_squre
    def SetSqureFromXY(self, _x,_y,_squre):
        _pos=_x+_y*self.width
        self.SqureContainer[_pos]=_squre
    def GetSpaceXY(self):
        x=self.spacePos%self.width
        y=(self.spacePos-x)/self.width
        return [x,y]
    def Play(self,action):
        [spaceX,spaceY]=self.GetSpaceXY()
        if action==Game.Up:
            if spaceY==0:
                return False
            else:
                space_squre=self.GetSqureFromXY(spaceX,spaceY)
                up_squre=self.GetSqureFromXY(spaceX,spaceY-1)
                self.SetSqureFromXY(spaceX,spaceY,up_squre)
                self.SetSqureFromXY(spaceX,spaceY-1,space_squre)
                self.spacePos=self.spacePos-self.width
                return True
        if action==Game.Down:
            if spaceY==self.height-1:
                return False
            else:
                space_squre=self.GetSqureFromXY(spaceX,spaceY)
                down_squre=self.GetSqureFromXY(spaceX,spaceY+1)
                self.SetSqureFromXY(spaceX,spaceY,down_squre)
                self.SetSqureFromXY(spaceX,spaceY+1,space_squre)
                self.spacePos=self.spacePos+self.width
                return True
        if action==Game.Left:
            if spaceX==0:
                return False
            else:
                space_squre=self.GetSqureFromXY(spaceX,spaceY)
                up_squre=self.GetSqureFromXY(spaceX-1,spaceY)
                self.SetSqureFromXY(spaceX,spaceY,up_squre)
                self.SetSqureFromXY(spaceX-1,spaceY,space_squre)
                self.spacePos=self.spacePos-1
                return True
        if action==Game.Right:
            if spaceX==self.width-1:
                return False
            else:
                space_squre=self.GetSqureFromXY(spaceX,spaceY)
                up_squre=self.GetSqureFromXY(spaceX+1,spaceY)
                self.SetSqureFromXY(spaceX,spaceY,up_squre)
                self.SetSqureFromXY(spaceX+1,spaceY,space_squre)
                self.spacePos=self.spacePos+1
                return True
    def GetState(self):
        state=[]
        for i in range(0, self.height):
            for j in range(0, self.width):
                state.append(self.GetSqureFromXY(j,i).index)
        return state
    def PrintState(self):
        for i in range(0,self.height):
            for j in range(0,self.width):
                if j!=self.width-1:
                    print(self.GetSqureFromXY(j,i).index,end='\t')
                else:
                    print(self.GetSqureFromXY(j, i).index, end='\n')
    def isSuccess(self):
        for pos in self.SqureContainer:
            squre=self.SqureContainer[pos]
            if squre.index!=pos:
                return False
        return False
    def InitGame(self,steps=100):
        for i in range(steps):
            action = random.randint(0, Game.Actions - 1)
            self.Play(action)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
import pandas as pd
from tensorflow import contrib
def NextBatch(sess,memory,size=30):
    randomChoice=memory.sample(frac=0.1)
    real_q=sess.run(y2,feed_dict={input:np.array(list(randomChoice['state2']))})
    maxq=np.max(real_q,axis=1)
    return randomChoice['state1'],randomChoice['action'],randomChoice['reward']+factor*maxq
if __name__=="__main__":
    width=6
    height=6
    game=Game(width,height)
    game.InitGame()
    memLen=100
    episode=100
    curEpisode=0
    factor=0.8
    ##
    memory=pd.DataFrame(columns=['state1','action','reward','state2'])
    for i in range(memLen):
        step={}
        step['state1']=game.GetState()
        action=random.randint(0,Game.Actions-1)
        if action==0:
            step['action']=[1,0,0,0]
        elif action==1:
            step['action']=[0,1,0,0]
        elif action==2:
            step['action']=[0,0,1,0]
        else:
            step['action']=[0,0,0,1]
        if game.isSuccess():
            reward=1
            step['reward']=reward
            step['state2']=None
            curEpisode=curEpisode+1
            game.InitGame()
        else:
            reward=0
            step['reward']=reward
            step['state2']=game.GetState()
        memory.loc[memory.shape[0]+1]=step
    inDim=game.count
    input = tf.placeholder('float',[None, inDim])
    w1 = tf.Variable(tf.random_uniform([inDim, 10], -1, 1))
    b1 = tf.Variable(tf.random_uniform([10], -1, 1))
    y1 = tf.nn.relu(tf.matmul(input, w1) + b1)
    w2 = tf.Variable(tf.random_uniform([10, 10], -1, 1))
    b2 = tf.Variable(tf.random_uniform([10], -1, 1))
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
    w3 = tf.Variable(tf.random_uniform([10, Game.Actions], -1, 1))
    b3 = tf.Variable(tf.random_uniform([Game.Actions], -1, 1))
    y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)


    action = tf.placeholder('float', [None,Game.Actions])
    qValue=tf.placeholder('float',[None,])
    coss=tf.reduce_mean(tf.reduce_sum(tf.multiply(y3,action),axis=1)-qValue,axis=0)
    #for i in range(0,)
    #split = tf.slice(y3,[0,action],[-1,1])
    #coss = tf.reduce_mean(tf.reduce_sum(tf.square(split-output), 1))
    train = tf.train.AdamOptimizer(1e-4).minimize(coss)
    init = tf.global_variables_initializer()
    g1 = tf.get_default_graph()
    sess = tf.Session(graph=g1)
    tf.summary.scalar('coss',coss)

    merged=tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('TensorBoard', sess.graph)

    sess.run(init)
    for i in range(10000):
        bx,_action, by = NextBatch(sess,memory)
        summary,_=sess.run([merged,train],feed_dict={input:bx,output:by,action:[0,0,0,1]})
        train_writer.add_summary(summary,i)
        if i % 50:
            print(sess.run(coss,feed_dict={input:bx,output:by,action:1}))
    train_writer.close()
    g2=tf.Graph()

    plotX=np.arange(-1,1,0.01)
    plotX=plotX.reshape(-1,1)
    plotY=np.power(plotX,2)
    _plotY=sess.run(split,feed_dict={input:plotX,output:plotY,action:1})
    fig1=plt.figure('fig1')
    plt.plot(plotX,plotY)
    plt.plot(plotX,_plotY)
    fig2=plt.figure('fig2')
    plotY = np.power(plotX, 2)
    _plotY2 = sess2.run(split, feed_dict={input: plotX, output: plotY, action: 1})
    plt.plot(plotX, plotY)
    plt.plot(plotX, _plotY2)
