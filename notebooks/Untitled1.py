
# coding: utf-8

# In[6]:

import fns
from fns import *
from fns.functionsTF import *
get_ipython().magic('matplotlib inline')


# In[7]:

dt = 0.1


# In[39]:

v = 0
I = 200
t_rest = 0
v_ = []
T=1000
for t in range(T):
    v = v + (dt/5* (-v + 0.15*I))*(t>t_rest)
    vv = v>25
    v = vv*(-30)+(1-vv)*v
    if vv:
        t_rest = (t + 30)*vv
    v_.append(v)
    
plt.plot(v_)


# In[38]:

v = 0
I = 200
t_rest = 0
v_ = []
T=1000
for t in range(T):
    v = v + (dt/10* (-v + 0.01*I))*(t>t_rest)
    vv = v>1.4
    v = vv*(0)+(1-vv)*v
    if vv:
        t_rest = (t + 30)*vv
    v_.append(v)
    
plt.plot(v_)


# In[ ]:



