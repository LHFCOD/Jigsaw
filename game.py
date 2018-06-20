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
        return True
    def InitGame(self,shuffle=True,steps=100):
        #如果打乱
        if shuffle:
            for i in range(steps):
                action = random.randint(0, Game.Actions - 1)
                self.Play(action)
        else:
            for i in range(self.height):
                for j in range(self.width):
                    squre = Squre()
                    squre.index = i + j * self.width
                    if i != 0 and j != 0:
                        squre.isSpace = False
                    else:
                        squre.isSpace = True
                    self.SqureContainer[squre.index] = squre

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
import pandas as pd
from tensorflow import contrib
import sys
import time
import math
def NextBatch(sess,memory,size=30):
    randomChoice=memory.sample(frac=0.1)
    real_q=sess.run(y3,feed_dict={input:np.array(list(randomChoice['state2']))})
    maxq=np.max(real_q,axis=1)
    return np.array(list(randomChoice['state1'])),np.array(list(randomChoice['action'])),np.array(list(randomChoice['reward']+factor*maxq))
def ActionToVector(action):
    vector=np.zeros([Game.Actions])
    vector[action]=1
    return vector
if __name__=="__main__":
    print("start time:%s"%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    width=4
    height=4
    game=Game(width,height)
    #不打乱游戏
    game.InitGame(shuffle=False)
    memLen=100
    episode=100
    curEpisode=0
    factor=0.8
    prob=0.9
    iterCount=10000
    perReward=100
    #设置最大步数，防止reward总为0
    maxStep=1000
    curStep=0
    #设置checkpoint步数
    checkStep=100
    checkpoint_dir="checkpoint"
    #设置是否从已有模型进行运行
    isLoad=True
    #从命令行获取迭代步数
    if len(sys.argv)==2:
        iterCount=int(sys.argv[1])
    ##
    memory=pd.DataFrame(columns=['state1','action','reward','state2'])
    for i in range(memLen):
        step={}
        step['state1']=game.GetState()
        action=random.randint(0,Game.Actions-1)
        game.Play(action)
        step['action']=ActionToVector(action)
        if game.isSuccess():
            reward=perReward
            step['reward']=reward
            step['state2']=game.GetState()
            curEpisode=curEpisode+1
            game.InitGame(shuffle=False)
        else:
            reward=0
            step['reward']=reward
            step['state2']=game.GetState()
        print(step)
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
    p1=tf.multiply(y3,action)
    p2=tf.reduce_sum(p1,axis=1)
    p3=tf.pow(p2-qValue,2)
    coss=tf.reduce_mean(p3,axis=0)
    train = tf.train.AdamOptimizer(1e-2).minimize(coss)
    init = tf.global_variables_initializer()
    tf.summary.scalar('coss',coss)

    g1 = tf.get_default_graph()
    sess = tf.Session(graph=g1)
    saver = tf.train.Saver()
    if isLoad:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    merged=tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('TensorBoard', g1)
    success_count=0


    sess.run(init)
    for i in range(iterCount):
        print("step:%s"%i)
        #summary,_=sess.run([merged,train],feed_dict={input:bx,output:by,action:[0,0,0,1]})
        #train_writer.add_summary(summary,i)
        ##游戏走步
        step = {}
        step['state1'] = game.GetState()
        nextQ=sess.run(y3,feed_dict={input:np.array(game.GetState()).reshape(1,-1)})
        nextAction=nextQ.argmax(axis=1)[0]
        realAction=nextAction
        nowProb=prob*math.log(2)/math.log(i+2)
        print("nowProb:%s"%nowProb)
        #探索
        if random.random()<0.5:
            randAction=random.randint(0,Game.Actions-2)
            if randAction >=nextAction:
               randAction=randAction+1
            realAction=randAction
        step['action']=ActionToVector(realAction)
        game.Play(realAction)
        #增加一步
        curStep=curStep+1
        if game.isSuccess():
            reward=perReward
            step['reward']=reward
            step['state2']=game.GetState()
            curEpisode=curEpisode+1
            game.InitGame(shuffle=False)
            success_count=success_count+1
            #重新计数

        else:
            reward=0
            step['reward']=reward
            step['state2']=game.GetState()
            if curStep>= maxStep:
                game.InitGame(shuffle=False)
                curStep=0
        memory.reset_index(drop=True,inplace=True)
        memory.drop([0],inplace=True)
        print(step)
        memory.loc[memory.shape[0] + 1] = step
        #梯度下降
        if i % 50 == 0:
            _state, _action, _q = NextBatch(sess, memory)
            print('coss:',end='')
            print(sess.run(coss,feed_dict={input:_state,qValue:_q,action:_action}))
            result=sess.run(merged,feed_dict={input:_state,qValue:_q,action:_action})
            train_writer.add_summary(result,i)
        if i % checkStep == 0:
            saver.save(sess,checkpoint_dir+"/model.ckpt",global_step=i)
    train_writer.close()
    print('success_count:%s'%(success_count))
    print('iterCount:%s'%(iterCount))
