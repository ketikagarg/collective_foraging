




Follow the instructions below to replicate findings from two papers:
1. Garg, K., Kello, C. T., & Smaldino, P. E. (2022). Individual exploration and selective social learning: balancing exploration–exploitation trade-offs in collective foraging. Journal of the Royal Society Interface, 19(189), 20210915.
2. Garg, K., Smaldino, P. E., & Kello, C. T. (2024). Evolution of explorative and exploitative search strategies in collective foraging. Collective Intelligence, 3(1), 26339137241228858.

    The main code for ABM is in _main_module.ipynb_

    Some data files for each parameter combination are in the folder 'data'. The code to replicate the figures in the main text is in the Jupyter Notebook called 'analysis'. The notebook reads in files from 'data'. 


    To visualize the model and generate output files:

    1. Access Binder link by clicking on the badge--> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ketikagarg/collective_foraging/HEAD)
    2. Wait for Binder to build the repository. It will automatically launch JupyterLab. 
    3. Click on 'Help' and select 'Launch Classic Notebook'. A new page will take you to a list of files.
    4. **Open _'Visualize.ipynb'_ and run it.** 



    ---------------------------------------------------------------------------------------------------------------------------------

 <b><u>To access code for the paper "Evolution of explorative and exploitative search strategies in collective foraging", please go to the folder "evolution/".</b></u>

NOTE: The evolutionary model operates in the same way as the 2022 model except the following changes:

1. In 2022 model, there is only one generation per simulation. In evolutionary model (henceforth called EM), there are 3000 generations per simulation run.
   
3. In EM, the function "crea_agents" generates agents with Levy walk exponents from a uniform distribution, s.t., the generated population starts as a mixed, heterogenous group containing of agents with all search strategies [1.1, 1.5, 2.0, 2.5, 3.0, 3.5]. The populations generated in the 2022 model started with a specific Levy exponent s.t. the agents in a group were homogenous in terms of their search strategy.
   
5. In EM, the following function simulates selection of agents between generations based on their fitness (here defined as search efficiency).
    "probabilistic_reproduction":
   - This function is called at the end of every generation and gets a list of all agents' Levy exponent and search efficiency. It then calculates the mean efficiency of each Levy exponent in the given population. Then, it calculates the proportional fitness of a Levy exponent by normalizing its associated mean fitness. The resulting normalized fitness of a Levy exponent is used as a probability to generate agents with that exponent.
   - After creating a new generation of agents, with a certain mutation rate ("mutrate"), it picks a random agent in the group and assigns a random Levy exponent to it. 
   - The function also resets agent locations, resource environment and agent efficiencies.
    
   
6. The folder "evolution/" also contains code for the ARS model. The ARS model is different only over one function ("Agent.update"):
   - In non-ARS model, if an agent is not on a resource (time, t), it decides whether to take a random walk or move towards a neighbor on a resource. In the ARS model, there is an additional step: if the agent was on a resource in the previous step (time, t-1) and now (time, t) is not --> it searches within its radius to find other resources and moves to one of them if finds a resource. If not, the agent decides whether to take a random walk or to move towards a neighbor.
   - See the highlighted parts in the ARS flowchart for the additional decision-step that the agent makes in ARS.  

