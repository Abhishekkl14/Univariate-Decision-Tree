#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Height Prediction


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#Creating a simple height dataset 
#Features: height of the person, labels: 0-short, 1-height

X = [[150], [155], [160], [165], [170], [175], [180], [185]]
y = [0, 0, 0, 0, 1, 1, 1, 1 ]


# In[3]:


#splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[4]:


clf = DecisionTreeClassifier(max_depth=1) #depth limit 1

#training teh decision tree

clf.fit(X_train, y_train)


# In[6]:


#predicting on the test set

y_pred = clf.predict(X_test)


#claculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the univariate decision tree: {accuracy:.2f}")

new_person_heigth = [[152]]
prediction = clf.predict(new_person_heigth )
print("Prediction(0=Short, 1=Tall):", prediction[0])


# In[21]:


#Fruit Classification


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt



# In[22]:


#Features: Weight of the fruit in grams

X = [[150], [160], [170], [180], [190], [200], [210], [220]]
y = [0,0,0,1,1,1,1,1]


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[24]:


clf = DecisionTreeClassifier(max_depth=1) #depth limit 1

#training teh decision tree

clf.fit(X_train, y_train)


# In[25]:


plt.figure(figsize=(8,6))
tree.plot_tree(clf, filled=True, feature_names=['Weight'], class_names=['Mango', 'Orange'], rounded = True, fontsize=12)


# In[27]:


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the univariate decision tree: {accuracy:.2f}")

new_fruit_weight = [[220]]
prediction = clf.predict(new_fruit_weight)
print("Prediction (0=Mango, 1=Orange):", prediction[0])  


# In[ ]:





# In[ ]:




