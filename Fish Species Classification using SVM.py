#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
fish=pd.read_csv('Fish.csv')
fish


# In[14]:


fish['Species'].unique()


# In[113]:


from sklearn.preprocessing import LabelEncoder as le


# In[114]:


le_Species=le()


# In[115]:


fish['Species_en']=le_Species.fit_transform(fish['Species'])


# In[116]:


fish


# In[117]:


fish['Species_en'].unique()


# In[118]:


fish[fish['Species_en']==0] #0 to 34


# In[119]:


fish[fish['Species_en']==1] # 35to 54


# In[57]:


fish1[fish1['Species_en']==2]


# In[120]:


fish[fish['Species_en']==3]


# In[121]:


fish[fish['Species_en']==4]


# In[122]:


fish[fish['Species_en']==5]


# In[123]:


fish[fish['Species_en']==6]


# In[124]:


fish11=fish[:35]
fish2=fish[35:55]
fish3=fish[55:61]
fish4=fish[61:72]
fish5=fish[72:128]
fish6=fish[128:145]
fish7=fish[145:]


# In[125]:


import matplotlib.pyplot as plt
plt.scatter(fish11['Height'],fish11['Width'],marker='^',color='green')
plt.scatter(fish2['Height'],fish2['Width'],marker='^',color='blue')
plt.scatter(fish3['Height'],fish3['Width'],marker='^',color='skyblue')
plt.scatter(fish4['Height'],fish4['Width'],marker='^',color='orange')
plt.scatter(fish5['Height'],fish5['Width'],marker='^',color='brown')
plt.scatter(fish6['Height'],fish6['Width'],marker='^',color='violet')
plt.scatter(fish7['Height'],fish7['Width'],marker='^',color='black')
plt.title('Heigth and Width of Fish')
plt.xlabel('Height')
plt.ylabel('Width')
plt.show()


# In[66]:


from sklearn.model_selection import train_test_split


# In[126]:


fish


# In[127]:


x=fish.drop(['Species','Species_en'], axis='columns')


# In[128]:


y=fish.Species_en


# In[129]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[130]:


x_train


# In[131]:


len(x_train)


# In[132]:


len(x_test)


# In[133]:


from sklearn.svm import SVC


# In[134]:


model=SVC()


# In[135]:


model.fit(x_train,y_train)


# In[136]:


model.predict(x_test)


# In[137]:


model.score(x_test,y_test)


# In[140]:


# Number 3 = Species of fish is Pike
model.predict([[2100,45,35,21,211,12]])


# In[ ]:




