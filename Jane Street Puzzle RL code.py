# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:47:31 2021

@author: Adam Keogh
""" 

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

random.seed(1)
lim =100 # running total limit to end the game
choicelim = 10 # max number that a player can choose
num_of_games = 100000 #how many training games will be played
num_of_test_games = 100  # the number of games we will test our player with
test_freq = 1000 # the frequency of testing during the training, i.e. every 1000 training games testing takes place
epsilon = 0.3 # the exploration parameter (how often player will take random move rather then move with highest Q-value)
gamma = 1  # discount rate          
alpha = 0.7 # learning rate

class Game:
    """This class contains the basic operations of the counting game, can be adjusted based on whether learning player is to go first or second, etc."""
    board =None
    board_lim=0 # this parameter will be the running total limit to end the game
    
    
    def __init__(self, board_lim=lim):
        # initialise the board
        self.board_lim = board_lim
        self.reset()
        
        
    def reset(self):
        # resets the board before each game, can be adjusted to have opponent go first
        self.board = 0
       
      
        
    def opponentplay(self, opponentchoice):
        # actions opponents move and updates board
        start_board = self.board + opponentchoice
        game_over = start_board >= self.board_lim
        if game_over:       
            self.board = self.board_lim 
            return game_over
        else:
            self.board = start_board
            return game_over    
        
    def ourplay(self, choice, oppQ):
        # makes a move given our choice, then makes opponents choice given opponent Q function
        # returns  : (reward, game ended by us?, game ended by opponent?, opp choice)
        new_board = self.board + choice
        if new_board < self.board_lim:#if our choice stays in bounds, set board and let opponent go
            self.board = new_board
            game_over = False
            #######################################################
            
            state=str(choice)+","+str(self.board) 
            
            options = list(oppQ.loc[oppQ[state]==oppQ[state].max()].T.columns) #lists the best moves according to q table
            if 11-choice in options:
                options.remove(11-choice)
            
            if options==[]:
                x = random.randint(1,choicelim)
                while x ==11-choice:
                    x = random.randint(1,choicelim)
            else:
                x = random.sample(options,1)[0]
    
              
            opp_game_over = self.opponentplay(x)
            #######################################################
            if opp_game_over:
                return (100, game_over, opp_game_over,x)
            else:
                return (0, game_over, opp_game_over,x) #continue on
        else:
            x=0 #just null value if i end game
            return (-100, True, False, x) #second one, did i end game, third did they end game
   
    
      
  
def playgame(epsil, test_Y_or_N, opp_Q, path_Y_or_N):
    ## this function plays through one whole game
    ## takes epsilon, a boolean representing whether it is a test game rather than a training game
    ## the opponents Q function, and a boolean representing whether we want to store game paths or not as inputs
    ## Returns reward of game if path_Y_or_N is False, otherwise returns (reward, path)
    
        game.reset() 
        total_reward = 0
        state=game.board
        game_over = False
        opp_game_over=False
        path=[]
    
        # decide on our player's choice to start game, using exploration
        if random.random()<epsil:
            choice = random.randint(1,choicelim)
            
        else:
            options = list(q_table.loc[q_table[state]==q_table[state].max()].T.columns) #lists the best moves according to q table
            choice = random.sample(options,1)[0] # randomly tiebreaks from these best moves
 
        reward, game_over, opp_game_over,opp_choice = game.ourplay(choice,opp_Q) # this actions our move and our opponents
        path.append(choice)
        path.append(choice+opp_choice)
        
        total_reward += reward
        if game_over or opp_game_over: # if our player has lost or opponent has lost, q(next state) used to update will be 0 as game will be in terminal state
            next_state_max_q_val = 0
        else: 
            next_state = str(opp_choice)+","+str(game.board) # pair together opponent move and running total of game as the state
            next_state_max_q_val = q_table[next_state].max() # take the max value of Q(next state) to use in update
        if not test_Y_or_N: 
            q_table.loc[choice,state] = q_table.loc[choice,state]+alpha*(reward + gamma * (next_state_max_q_val -q_table.loc[choice,state])) # Q-table update rule, dont update Q when testing
        
        # subsequent gameplay after initial moves, same steps repeated on loop until game ends
        while not game_over and not opp_game_over:
            state=str(opp_choice)+","+str(game.board)
        
            if random.random()<epsil:
                choice = random.randint(1,choicelim)
                while choice == 11-opp_choice:
                    choice = random.randint(1,choicelim)
            
            else:
                options = list(q_table.loc[q_table[state]==q_table[state].max()].T.columns) 
                if 11-opp_choice in options:
                    options.remove(11-opp_choice)
            
                if options==[]:
                    choice = random.randint(1,choicelim)
                    while choice ==11-opp_choice:
                        choice = random.randint(1,choicelim)
                else:
                    choice = random.sample(options,1)[0] 
           
            reward, game_over, opp_game_over, opp_choice = game.ourplay(choice,opp_Q) 
            path.append(choice+int(path[-1]))
            path.append(opp_choice+int(path[-1]))
            
            total_reward += reward
            if game_over or opp_game_over:
                next_state_max_q_val = 0
            else:
                next_state =str(opp_choice)+","+ str(game.board) # this should always be in bounds because of if statement
                next_state_max_q_val = q_table[next_state].max()
            if not test_Y_or_N:
                q_table.loc[choice,state] = q_table.loc[choice,state]+alpha*(reward + gamma * (next_state_max_q_val -q_table.loc[choice,state]))
        path.append(total_reward)
        if path_Y_or_N==True:
            return (total_reward,path)
        else:
            return (total_reward)           


# now to simulate game, first thing is to set up empty Q table
lis=[0]
for j in np.arange(1,lim,1):
    for i in np.arange(1,choicelim+1,1):
        if i <= j: 
           lis.append(str(i)+","+str(j))
#################################################################################
q_table = pd.DataFrame(0, index=np.arange(1,choicelim+1,1), columns=lis) # Q-table filled with 0s
#################################################################################

r_list = [] # used to track rewards
oppQ = q_table # this means opponent will use same Q-table as player to choose moves. Alternatively could iteratively improve opponent Q-table
game = Game() # instance of class Game

test_results = [] # track testing results


for g in range(num_of_games):
    ### mid-training tests
    if g%test_freq==0:
        print(g)
        counter=0
        for i in range(num_of_test_games):
            reward = playgame(0,True,oppQ,False) # epsilon 0 so no exploration during testing, playing against self
            if reward==100:
                counter+=1
        test_results.append(counter/num_of_test_games) # add proportion of games won to the list
    ###################################################    
        
    result = playgame(epsilon,False,oppQ,False) # play training game
    r_list.append(result)

### final mid-training test and plot pf results
counter=0
for i in range(num_of_test_games):
    reward = playgame(0,True,oppQ,False)
    if reward==100:
        counter+=1
test_results.append(counter/num_of_test_games)
    
plt.figure(figsize=(14,7))
plt.plot(range(int(num_of_games/test_freq+1)),test_results, marker='o', linestyle='dashed')
plt.xlabel('Test number')
plt.ylabel('Proportion won')
plt.title('Result of mid-training tests')
plt.show()  
####################################

#Path creation for small number of games
pathlist=[]
for g in range(100):
    result,path =playgame(0,True, oppQ, True)
    pathlist.append(path)
#############################

## Post training testing  
final_test_results = []
randtest_results = []

# Test player against themself many more times to ensure always win
for g in range(10000):
    result = playgame(0,True, oppQ,False)
    final_test_results.append(result)
plt.figure(figsize=(14,7))    
plt.plot(range(10000),final_test_results)
plt.show()

# Test player against opponent who makes random moves
oppQ =pd.DataFrame(0, index=np.arange(1,choicelim+1,1), columns=lis)
for g in range(1000):
    result = playgame(0,True, oppQ,False)
    randtest_results.append(result)
plt.figure(figsize=(14,7))    
plt.plot(range(1000),randtest_results)
plt.show()