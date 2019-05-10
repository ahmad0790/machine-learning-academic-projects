1) To run the jupyter notebooks please install conda and create a new environment

2) The code is available on my GitHub repo here as jupyter notebooks.
https://github.gatech.edu/akhan361/ML_Project_4

3) Then install the files in the requirements.txt file in the GitHub repo. Some of these are not needed for the assignment but the key libraries that you should install install are Pandas, Matplotlib, Seaborn, Sklearn, Numpy, Jupyter,gym without which the code will not run.

Is is especially essential that you install gym since the MDP is based on a gym framework.

4) Please make sure to also download the MDPs. There are three MDPs. frozen_lake.py,  frozen_lake_rewards.py,  frozen_lake_rewards_v2.py.

The latter two MDPs have been modified with reward shaping.

4) Run the code through Jupyter notebook. The three notebooks are as follows:

1) RL Value vs Policy Iteration.ipynb
This notebook compares the performance of VI vs PI

2) RL VI and PI Experiments .ipynb
This notebook does additional experiments on the MDP by changing MDP attributes and parameters.

3) RL Q Learning Performance.ipynb
This notebook runs the Q learning algorithm for comparison to VI and PI.

Note: Special credit to these people who's implementations of VI, PI and QL I used. I copied their implementations and want to give credit where it is due. However all analysis is my own.

Denny Britz Github Repo
https://www.google.com/search?q=denny+britz+rl+github&oq=denny+britz+rl+github&aqs=chrome..69i57.6614j0j7&sourceid=chrome&ie=UTF-8

Moustafa Alzantot's Medium Blog
https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

