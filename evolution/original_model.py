#!/usr/bin/env python
# coding: utf-8

# Import dependencies

from __future__ import division

import random
import math
import os
# import time
# import random
import shutil
import numpy as np
from scipy import spatial as st 
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
from periodic_kdtree import PeriodicCKDTree
from sklearn.cluster import DBSCAN
from scipy import stats as sts
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pylab as pl
# from IPython import display
import sys
import copy
import itertools


# Set global parameters


digits = 3 ##the resolution for grid

epsilon = 0.001  ###
bounds = np.array([1, 1])  ## x,y length of the grid  
minstep = 0.001
generations = 3000


##enter the group size, patch numbers and alpha you want to test 
groupsize = int(sys.argv[1])
patches = int(sys.argv[2])
al = int(sys.argv[3])
beta = 3 
radius = 1  ##the radius within which agents can detect others



# Grid class which represents the environment on which simulation takes place. It also generates targets and agents. 




class Grid:

    def __init__(self, radius, patches, agents, beta, al):

        # This is what builds the grid. Called every time a new grid is constructed
        # initialize the variables
        self.num_agents = agents  ## number of agents 
        self.num_targets = patches  ##number of targets
        self.beta = beta ###target distribution given Beta
        self.alpha = al ## the fixed value of alpha or the social selectivity parameter
        ##create targets 
        self.crea_targets()
        ###create agents with fixed alpha and randomly distributed mu values  
        self.crea_agents()
        
        ##the radius of interaction for agents --> set to 1 
        self.rad = radius        
        
        ##initialize a parameter tgat checks how many targets have been consumed 
        self.deleted_targets = 0 


        #----------------------------------------------------------------------------------------------------------

    ###UPDATE:- runs agents at every time step. After agents update their position at every time step, CLUSTERING calculates the grouping between agents.
            
    def update(self):

        z=0
        r = self.rad
        while z < len(agents_list):
            agents_list[z].update(agents_list, r)
            
            z = z + 1
      
        
        #----------------------------------------------------------------------------------------------------------  

    ##CREATE TARGETS using power-law distribution       
    
    
    def crea_targets(self):
        self.size = 1
        # This method is in charge of creating the targets according to a resource distribution
        global targets_list
        targets_list =[]
        all_x = []
        all_y = []
        ##initialize seed population 
        for i in range(1,21) :  
            x = np.round(random.uniform(0,1)% self.size,digits)
            y = np.round(random.uniform(0,1)% self.size,digits)
            all_x.append(x) 
            all_y.append(y) 
            num = i 

            targets_list.append(Patch(x,y, num))
#         num= 0 
        self.make_targets(all_x, all_y, num)
        coord_targets = np.column_stack((all_x, all_y))
        ##create a KD Tree which contains all the targets 
        Grid.kdtree = PeriodicCKDTree(bounds, coord_targets)
        Grid.deleted_targets = 0 
        Grid.cluster_centroid = np.empty((0,2), float)
        Grid.cluster_time = 0 

    def spawntargets(self) :
        ##pick an already existed target from the list and use it as a seed to generate another target
        i = random.choice(range(len(targets_list))) 
        seeder_x = targets_list[i].x
        seeder_y = targets_list[i].y
        theta = random.uniform(0,360)
        theta = theta * (math.pi / 180)
        dis = self.levy()

        x =  np.round((seeder_x + (dis * np.cos(theta))) , digits)
        y = np.round((seeder_y + (dis * np.sin(theta))), digits )
        
        ##adjust for periodic boundaries
        if x > 1 :
            x = abs(1 - x)
        if y > 1 :
            y = abs(1 - y)
        if x < 0 :
            x = abs(1 - abs(x))
        if y < 0  :
            y = abs(1 -  abs(y)) 

        return x, y
#         

    def levy(self):
        xf = 0.1
        xi = 0.001
        mu = -self.beta
        m = mu + 1
        x = ((((xf ** m) - (xi ** m)) * (random.uniform(0,1)) ) + (xi ** m)) ** (1/m)
            
        return x 
        
    def make_targets(self, all_x, all_y, num) :
        num = num + 1
        ##continue generating targets until all the required number of targets is not achieved. 
        while len(targets_list) < self.num_targets+1 :
            x, y = self.spawntargets() 
            all_x.append(x) 
            all_y.append(y)             
            targets_list.append(Patch(x,y, num))
            num = num + 1
            
            if len(targets_list) == self.num_targets :
                break       

            
    #----------------------------------------------------------------------------------------------------------
    
    ##The following function creates agents.  
    
    
    def crea_agents(self):
        ##creates agents at random locations, with 0 energy, alpha, levy
        z = 1
        global agents_list 
        agents_list =[]

        ###these are the Levy walk alleles considered 
        self.mu_alleles = [1.1, 1.5, 2, 2.5, 3, 3.5 ]
        self.prob_mu_alleles = [1 / 6] * 6 


        while len(agents_list) < self.num_agents :
            x = random.uniform(0,1)
            y = random.uniform(0,1)
            levy = np.random.choice(self.mu_alleles, p=self.prob_mu_alleles)
            alpha = self.alpha
            num = z ### agent's ubique ID 
            agents_list.append(Agent(num,x,y,levy,alpha))
            z = z +1 

        ###shuffle agent list 
        random.shuffle(agents_list)

        #----------------------------------------------------------------------------------------------------------

    ###The following function finds if an agent is located on a target
    
    def findwhichtarget(self):
        # This method helps locate a target object based on its coordinates: x, y
        
        dis, indices = Grid.kdtree.query([self.x,self.y], k=1000, distance_upper_bound = minstep)
        ###if any target is not present --> return None
        if np.isfinite(dis).any() == False:
            return None 
        
        else:
            size = np.count_nonzero(~np.isinf(dis))
            ###if more than one targets present
            if size > 1 : 
                k = 0 
                ###check which target which has food left on it.
                while k < size:
                    i = indices[k]
                    ###if a target is found with food, deplete its food and add it to the list of deleted targets
                    #####return the target ID which was consumed 
                    if targets_list[i].food == 1 :
                        targets_list[i].food = 0 
                        Grid.deleted_targets += 1 
                        return targets_list[i].num
                    k = k + 1
                ####if none of the targets have any food left, return None
                return None
            ###if only on target is present, then check if there is food on it. 
            else:
                if targets_list[0].food == 1 :
                    targets_list[0].food = 0 
                    Grid.deleted_targets += 1
                    return targets_list[0].num
                else:
                    return None


##------------------------------------------------------------------------------------------------
###the following function is the genetic algorithm 

    def probabilisitic_reproduction(self):

        ff=[]
        nums=[]
        i = 0
        j = 0 
        ff_die = []
        nums_die = []
        ff_all = []
        al = []
        mu = []
        global agents_list
                
        mu = [c.mu for c in agents_list]
        al = [c.alpha for c in agents_list]
#         ff = [c.nums for c in agents_list]
        
        ##get the efficiency of all agents 
        ff = []
        for i in [1.1, 1.5, 2, 2.5, 3, 3.5]:
            mulist = fitness_mu(self,i)
            ff.append(np.mean(mulist))
            
        ##get the corresponding mu values 
        ff = np.array(ff)
        nans = np.where(np.isnan(ff))
        new_mulist = np.array([1.1, 1.5, 2, 2.5, 3, 3.5])[tuple(nans)]
        
        ##normalize the efficiencies
        ff = ff[~np.isnan(ff)]
        weights=[]
        b = (ff - min(ff)) / (max(ff) - min(ff))
        b /= sum(b)
        
        # print(ff)
        # print(nans)
        # print(new_mulist)
        # print("b", b)

        ##create a new generation of agents with values of mu dependent on their parents.
        ### plus, intialize them randomly 
        i=0
        global gen_counter
        gen_counter = gen_counter + 1 
        agents_list = []
        for i in range(self.num_agents):
            levy = np.random.choice(new_mulist, p=b)
            ##find the parent --> copy genetic data 
            alpha = self.alpha 
            x = random.uniform(0,1)  
            y = random.uniform(0,1)
            agents_list.append(Agent(i,x, y,levy,alpha))

                
        ###regenerate environment
        self.crea_trees()
        random.shuffle(agents_list)               
        
        ##select one agent randomly and mutate its inherited allele 
        mutrate = 0.05
        chance = random.uniform(0,1) 
        if mutrate > chance:
            mutant = np.random.choice(range(self.num_agents))
            agents_list[mutant].mu = np.random.choice(self.mu_alleles)        
#             print("mutated")        


    def fitness_mu(self, muval):
        muval_list = []
        for c in agents_list:
            if c.mu == muval:
                muval_list.append(c.eff)
            
        return muval_list


#----------------------------------------------------------------------------------------------------------------------------##

        
    def periodic_distance(self, X):
        L = 1 
        for d in range(X.shape[1]):
            # find all 1-d distances
            pd=st.distance.pdist(X[:,d].reshape(X.shape[0],1))
            # apply boundary conditions
            pd[pd>L*0.5]-=L

            try:
                # sum
                total+=pd**2
            except :
                # or define the sum if not previously defined
                total=pd**2
        # transform the condensed distance matrix...
        total=pl.sqrt(total)
        # ...into a square distance matrix
        square=st.distance.squareform(total)
#         squareform(total)
        return square


#-------------------------------------------------------------------------------------------------------------------------
    ### the following function generate output files:

    
    def output(self,i):
                
        avg_eff = np.mean([c.eff for c in agents_list])
        avg_dis = np.mean([c.tdis for c in agents_list])
        avg_en = np.mean([c.food for c in agents_list])

        ###find proportions of agents with the different mu alleles in a population 
        search = [1.1, 1.5, 2, 2.5, 3, 3.5]
        u, c = np.unique([c.mu for c in agents_list], return_counts = True)
        freq = [c[u.tolist().index(i)] if i in u else 0 for i in search]

        stringdis =  str(avg_en) + "," + str(avg_dis) + "," + str(avg_eff) +  "," + str(self.list_format([c.mu for c in agents_list])) + "," + str(self.list_format([c.alpha for c in agents_list])) + "," + str(self.list_format([c.eff for c in agents_list])) + "," + str(self.list_format(freq)) +  "\n"


        
        # stringdis =  str(avg_en) + "," + str(avg_dis) + "," + str(avg_eff) +  "," + str(avg_size) + "," + str(avg_time) + "," + str(all_clusters) + "," + str(avg_simul_clusters) + "\n"
        return stringdis





class Patch:
    # This class represents a target with all functions and attributes
    
    # create a target with a position, and a name
    def __init__(self, x, y, num):
        # This method is called for creating a new patch
        self.x = x
        self.y = y
        self.num = num
        self.food = 1


# 

# In[26]:


class Agent:

    def __init__(self, num, x,y, levy,alpha):
        
        
        self.num = num  #ID
        self.x = x ##x-coordinate
        self.y = y  ##y-coordinate
        self.mu = levy   ###the levy exponent for individual search
        self.alpha = alpha ####social learning component 
        
        self.food = 0 ### food found so far
        self.target = 0  ###target on which the agent is on 
        
        self.eff = 0 ###search efficiency
        
        
        self.rw_switch = 0 ###conducting a random walk or not-- boolean switch
        self.tw_switch = 0 ###conducting a targeted walk towards another agent or not -- boolean switch 
        self.d = 0 
        self.theta = 0 
        self.tdis = 1
        
        
        #----------------------------------------------------------------------------------------------------------

    def update(self, agents_list,r):  ##check if on a target or not

        ##check if the agent is on a target
        num = Grid.findwhichtarget(self)

        ###update search efficiency at every time-step
        if self.food == 0 :
            self.eff = 0
        else : 
            self.eff = self.food / self.tdis
            
        
        
        ###if there is target on current location:
        if num != None:            
            self.target = num   ###update which target the agent is on 
            self.food = self.food + 1  ##update the total number of targets/food found
            
            ### terminate both walks 
            self.rw_switch = 0 
            self.tw_switch = 0 
            self.walk = 0 


    
            
            
        # If there is no food, then look for others 
        #if P(scrounge) = 1, if rw, terminate rw walk and start a targeted walk towards neighbor
        # if P(scrounge), if tw, terminate current walk and start moving in a new direction towards another neighbor 
        #if P(scrounge) = 0 , if rw, continue 
        ## if P(scrounge) = 0, if tw, continue
        
        else : 
            ###look for others
            self.target = 0
            #####if already walking towards another agent
            if self.tw_switch == 1 :
                newtarget = self.check_forothers(agents_list,  r)
                ####if new agent is detected, start a new TW
                if newtarget != None : 
                    ###if distance to new agent is more than 0, take a step
                    if self.d > 0 :
                        self.step()
                        ####if after taking a step, agent has reached the destination, terminate the new TW
                        if self.d <= 0 :
                            self.tw_switch = 0
                    ###if the distance to another agent is 0, terminate the new TW       
                    else : 
                        self.tw_switch = 0 
                
                
                
                #######if no new agent is detected, continue previous TW
                else :   
                    if self.d > 0 :
                        self.step()
                        if self.d <= 0 :
                            self.tw_switch = 0
                    else : 
                        self.tw_switch = 0 
                            
            
            #######if the agent was not doing a TW or RW    
            elif self.tw_switch == 0 and self.rw_switch == 0 :
                ###check for other agents 
                newtarget = self.check_forothers(agents_list, r)
                ####if an agent is detected, start a TW 
                if newtarget != None :   
                    self.tw_switch = 1
                    self.rw_switch = 0 
                    if self.d > 0 :
                        self.step()
                        if self.d <= 0 :
                            self.tw_switch = 0
                    else : 
                        self.tw_switch = 0 
                        
                ####if no new agent is detected, start a random walk (RW)
                else : 
                    rw_update = self.rw()  ##start a rw
                    self.d = rw_update[0]  ###get the distance to move
                    self.theta = rw_update[1]  ### get the direction 
                    if self.d > 0 :
                        self.step()
                        self.rw_switch = 1
                        if self.d <= 0 :
                            self.rw_switch = 0
                    else : 
                        self.rw_switch = 0 
                            
                    
            #####if the agent was already doing a random walk 
            elif self.rw_switch == 1 : 
                ###check for other agents 
                newtarget = self.check_forothers(agents_list, r)
                ###if a new agent deteced, terminate RW and start TW 
                if newtarget != None :   ##stop rw and do tw instead
                    self.tw_switch = 1
                    self.rw_switch = 0 
                    if self.d > 0 :
                        self.step()
                        if self.d <= 0 :
                            self.tw_switch = 0
                    else : 
                        self.tw_switch = 0 
                
                ####Else if no agent detected, continue the random walk 
                else : 
                    if self.d > 0 :   ##continue rw
                        self.step()
                        if self.d <= 0 :
                            self.rw_switch = 0
                    else : 
                        self.rw_switch = 0 
                        
            
    
            

            
            
    ####this function calculates the distance for a random walk based on Levy distribution 
    def calc_dist(self):
        xf = 1
        if self.mu == 'random' :
            x = minstep
            return x 
        elif self.mu == 'straight':
            x = 1000000000 
            return x 
    
        else:
            xi = minstep
            mu = -self.mu
            m = mu + 1
            x = ((((xf ** m) - (xi ** m)) * (random.uniform(0,1)) ) + (xi ** m)) ** (1/m)
            if self.mu == 'random' :
                x = minstep
            return x
 
 
      #----------------------------------------------------------------------------------------------------------
     
    def rw(self) :     
        r = self.calc_dist() 
        theta = random.uniform(0,360)
        return( r, theta)

    ###this function updates agent's position,one step at a time 
    def step(self) : 
        
        if self.d <= minstep :
            r = self.d
        else :
            r = minstep
        theta = self.theta * (math.pi / 180)
        x2 = (self.x + (r * np.cos(theta)))
        y2 = (self.y + (r * np.sin(theta)))
        ###adjust for periodic boundaries
        if x2 >= 1 :
            x2 = abs(1 - x2)
        if y2 >= 1 :
            y2 = abs(1 - y2)
        if x2 <= 0 :
            x2 = abs(1 - abs(x2))
        if y2 <= 0  :
            y2 = abs(1 -  abs(y2))     
        self.x = x2
        self.y = y2
        ###update how much distance is left to move
        self.d = self.d - r
        #####update the total distance traveled by an agent 
        self.tdis = self.tdis + r

        
    #### this function checks for agents that are on a target and generates a TW based on alpha 
    
    def check_forothers(self, agents_list,  r):
        z = 0 
        all_x = []
        all_y = []
        creepable = []
        coord_others = []
        indices =[]
        i = 0 
               
         ###get other agents who are on a target and not yourself 
        
        for i in range(len(agents_list)):  
            if agents_list[i].target != 0 :
                creepable.append(i)
                
                    
                
        ###if there are agents that are on a target 
        if len(creepable) > 0 : 
            z = 0 
            for z in creepable :
                all_x.append(agents_list[z].x) 
                all_y.append(agents_list[z].y) 
                
            ###create a matrix for their coordinates
            coord_others = np.column_stack((all_x, all_y))
            
            ###find which agent is the closest
            dis, indices = self.distance(coord_others, (self.x, self.y), r)
            ###if no agent is present or outside radius, return None 
            if math.isinf(dis) == True :
                return None 
            ####else 
            else :
                dis2 = (dis *1000)
                chance = random.uniform(0,1)         
                ni= creepable[indices]   
                ####if already doing a TW 
                
                if self.tw_switch == 1 :
                ###check if current distance to previous agent is more than the distance to new agent 
                    if self.d > dis:
                        ##get the coordinates of the new agent 
                        new_coord = (agents_list[ni].x, agents_list[ni].y)
                        self.d = dis ###update distance for walk 
                        self.theta = self.tw_angle(self.x, self.y, new_coord[0], new_coord[1]) ###update angle for walk 
                        to_return = new_coord[0],new_coord[1], dis
                    else : 
                        return None 
                ###if currently not doing a TW 
                else:
                    if chance < math.exp(- self.alpha * dis2) :
                        new_coord = (agents_list[ni].x, agents_list[ni].y)
                        self.d = dis 
                        self.theta = self.tw_angle(self.x, self.y, new_coord[0], new_coord[1])
                        to_return = new_coord[0],new_coord[1], dis
                        return to_return
                    else : 
                        return None
            
        else : 
            return None
   


    def tw_angle(self, x1, y1, x2, y2):
        ###calculate angle between two agents 
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx > 0.5 :
            if x1 > 0.5 :
                x2 = 1 + (x2 - 0)
            else :
                x2 = 0 - (1 - x2)
        if dy > 0.5 :
            if y1 > 0.5 :
                y2 = 1 + (y2 - 0)
            else :
                y2 = 0 - (1 - y2) 
        dx = x2 - x1
        dy = y2 - y1
        theta = math.degrees(math.atan2(dy, dx))
        if (theta < 0.0) :
            theta += 360.0
        return theta
        

        
    def distance(self, x0, x1, upperbound):
        ####this function checks if the agents deteccted are within radius 
        delta = np.abs(x0 - x1)
        #####adjust for periodic boundaries
        delta = np.where(delta > 0.5 * bounds, delta - bounds, delta)
        alld =  np.sqrt((delta ** 2).sum(axis=-1))
        alld = list(map(lambda x: np.inf if x > upperbound else x, alld))
        ####return the distance to the closest agent and the ID of the closest agent 
        return np.min(alld), np.argmin(alld)

        
        
           
                
        

def main():    
    global gen_counter 
    global food_found
    global avg_sep 
    avg_sep = 0 

    gen_counter = 0 
    food_found = 0 
    
    
    grid = Grid(radius, patches, groupsize, beta, al)
    i = 1
    # Main cycle of each iteration
    string2=""
    all_string2 = ""
    gen_counter = 0
    global gen_counter 
    
    pf = 0 
    while gen_counter < generations:
        while pf < 30 : 
            pf = ( Grid.deleted_targets / (patches * 0.01))
            grid.update()
            i = i + 1
            
        ###when the simulation is done, get the result 
        string = grid.output(i)
        
        string2 = str(gen_counter) + "," + str(i) + "," +str(patches)+","+str(groupsize)+","+str(beta)+","+str(radius)+","+str(l) + "," + str(a) + "," + string
        all_string2 = all_string2 + string2
        grid.probabilisitic_reproduction()

    # file_name = l
    return string2







###call main function 
if __name__ == "__main__":
    main()

