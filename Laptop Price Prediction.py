#!/usr/bin/env python
# coding: utf-8

# In[1]:


#packages
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os


# In[2]:


data = pd.read_csv("laptop_data.csv")
data


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data.describe()


# In[6]:


data


# In[7]:


df = data[["Company","TypeName","Inches","ScreenResolution","Cpu","Ram","Memory","Gpu","OpSys","Weight","Price"]]


# In[8]:


df


# In[9]:


df.duplicated().sum()


# In[10]:


cat_col = [i for i in df.columns if df[i].dtype == 'O']
num_col = [i for i in df.columns if i not in cat_col]


# In[11]:


df[cat_col]


# In[12]:


df[num_col]


# In[13]:


for i in df[cat_col]:
    a = df[i].unique()
    print(a)


# In[14]:


df["Company"].unique()


# In[15]:


for i in df[cat_col]:
    a = df[i].unique()
    print(a)


# In[16]:


#converting 
df["Ram"] = df["Ram"].str.replace("GB","")
df["Weight"] = df["Weight"].str.replace("kg","")


# In[17]:


df["Ram"] = df["Ram"].astype("int32")
df["Weight"] = df["Weight"].astype("float32")


# In[18]:


df.info()


# In[19]:


df[num_col].corr()


# In[20]:


#EDA
#distribution of Price
print(sns.distplot(df["Price"],color ="red"))


# In[21]:


print(sns.boxplot(df["Price"],color="red"))


# In[22]:


#plotting cat
def drawplot(col):
    plt.figure(figsize =(15,7))
    sns.countplot(df[col],palette='plasma')
    plt.xticks(rotation="vertical")
    
toview = ["Company","TypeName","OpSys","Cpu"]
for col in toview:
    drawplot(col)
    


# In[23]:


cat_col


# In[24]:


plt.figure(figsize=(15,7))
sns.barplot(x =df["Company"],y = df["Price"])
plt.xticks(rotation="vertical")
plt.show()


# In[25]:


#laptop type and variation about price
sns.barplot(x = df["TypeName"], y =df["Price"])
plt.xticks(rotation="vertical")


# In[26]:


plt.scatter(df["Inches"],df["Price"])
plt.xlabel("Inches")
plt.ylabel("Price")


# In[27]:


df["ScreenResolution"].value_counts()


# In[28]:


df["Touchscreen"] = df["ScreenResolution"].apply(lambda element:1 if "Touchscreen" in element else 0)


# In[29]:


df["Touchscreen"].value_counts()


# In[30]:


sns.countplot(df["Touchscreen"],palette="plasma")


# In[31]:


#touchscreen in comparison of price of laptop
sns.barplot(x = df["Touchscreen"],y=df["Price"])


# In[32]:


#Creating new col IPS
df["IPS"] = df["ScreenResolution"].apply(lambda element:1 if "IPS" in element else 0)


# In[33]:


df["IPS"].value_counts()


# In[34]:


#IPS in comparison of price of laptop
sns.barplot(x = df["IPS"],y=df["Price"])


# In[35]:


splitdf = df["ScreenResolution"].str.split("x",n=1,expand=True)
splitdf.head()


# In[36]:


df["x_res"] = splitdf[0]
df["y_res"] = splitdf[1]


# In[37]:


df["x_res"] = df["x_res"].str.replace(",","").str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[38]:


df["x_res"] = df["x_res"].astype("int")
df["y_res"] = df["y_res"].astype("int")


# In[39]:


df


# In[40]:


#coorelation
#correlation
plt.figure(figsize = (15,10))
sns.heatmap(data = df.corr(),cmap='plasma',annot=True,linewidths=0.5)
plt.show()


# In[41]:


df.corr()["Price"]


# In[42]:


#joining both XandY RES
df["PPI"] = (((df["x_res"]**2+df["y_res"]**2))**0.5/df["Inches"]).astype("float")


# In[43]:


df


# In[44]:


df.corr()["Price"]


# In[45]:


df.drop(columns = ["x_res","y_res","Inches","ScreenResolution"],inplace=True)


# In[46]:


df


# In[47]:


#now we work on CPU Column
df["Cpu"].value_counts()


# In[48]:


text = "Intel Core i5 7200U 2.5GHz"
" ".join(text.split()[:3])


# In[49]:


df["CPU_name"] = df["Cpu"].apply(lambda text:" ".join(text.split()[:3]))


# In[50]:


df.head()


# In[51]:


def processortype(text):
    
    if text=="Intel Core i7" or text=="Intel Core i5" or text=="Intel Core i3":
        return text
    else:
        if text.split()[0] == "Intel":
            return "Other Intel Processor"
        else:
            return "AMD Processor"
df["CPU_name"] = df["CPU_name"].apply(lambda text:processortype(text))


# In[ ]:





# In[52]:


df


# In[53]:


sns.countplot(df["CPU_name"],palette = 'plasma')
plt.xticks(rotation='vertical')


# In[54]:


sns.barplot(df["CPU_name"],df["Price"])
plt.xticks(rotation="vertical")


# In[55]:


df["CPU_name"].value_counts()


# In[56]:


df.drop(columns = ["Cpu"],inplace = True)


# In[57]:


df


# In[58]:


#we check with ram 
sns.countplot(df["Ram"],palette = 'plasma')
plt.xticks(rotation='vertical')


# In[59]:


sns.barplot(df["Ram"],df["Price"])
plt.xticks(rotation="vertical")


# In[60]:


#Memory 
df["Memory"].value_counts()


# In[61]:


df["Memory"] = df["Memory"].astype(str).replace('\.0','',regex =True)

#replacce GB word with " "
df["Memory"] = df["Memory"].str.replace('GB','')

# replace the TB with "000"

df["Memory"] = df["Memory"].str.replace('Tb','000')

#split the word across the '+'

newdf = df["Memory"].str.split("+",n=1,expand=True)

newdf


# In[62]:


#will eleminate whitw spaces
df["first"] = newdf[0]
df["first"] = df["first"].str.strip()
df


# In[63]:


def applychanges(value):
    df["Layer1"+value] = df["first"].apply(lambda x:1 if value in x else 0)

listtoapply = ["HOD","SSD","Hybrid","FlashStorage"]
for value in listtoapply:
    applychanges(value)


# In[64]:


#remove all character
df["first"] = df["first"].str.replace(r'\D','')
df["first"].value_counts()


# In[65]:


df["Second"] = newdf[1]
df


# In[66]:


def applychanges(value):
    df["Layer2"+value] = df["Second"].apply(lambda x:1 if value in x else 0)

listtoapply = ["HOD","SSD","Hybrid","FlashStorage"]
df["Second"] = df["Second"].fillna("0")
for value in listtoapply:
    applychanges(value)
    
df["Second"] = df["Second"].str.replace(r'\D','')
df["Second"].value_counts()


# In[67]:


df["first"] = df["first"].astype("int")
df["Second"] = df["Second"].astype("int")
df


# In[68]:


#multiplying the elements
df["HOD"] = (df["first"]*df["Layer1HOD"]+df["Second"]*df["Layer2HOD"])
df["SSD"] = (df["first"]*df["Layer1SSD"]+df["Second"]*df["Layer2SSD"])
df["Hybrid"] = (df["first"]*df["Layer1Hybrid"]+df["Second"]*df["Layer2Hybrid"])
df["Flash_Storage"] = (df["first"]*df["Layer1FlashStorage"]+df["Second"]*df["Layer2FlashStorage"])


# In[69]:


df.drop(columns=["first","Second","Layer2HOD","Layer1HOD","Layer2SSD","Layer1SSD","Layer2Hybrid","Layer1Hybrid","Layer2FlashStorage","Layer1FlashStorage"],inplace=True)


# In[70]:


df


# In[71]:


df.drop(columns = ["Memory"],inplace=True)


# In[72]:


df


# In[73]:


df.corr()["Price"]


# In[74]:


df.drop(columns = ["HOD","Hybrid","Flash_Storage"],inplace =True)


# In[75]:


df


# In[76]:


#GPU
df["Gpu"].value_counts()


# In[77]:


#Extracting the brands
a = df["Gpu"].iloc[1]
print(a.split()[0])


# In[78]:


df["Gpu brand"] = df["Gpu"].apply(lambda x:x.split()[0])
sns.countplot(df["Gpu brand"],palette = "plasma")


# In[79]:


df = df[df["Gpu brand"]!= "ARM"]


# In[80]:


df= df.drop(columns = ["Gpu"])


# In[81]:


df


# In[82]:


df["Gpu brand"].value_counts()


# In[83]:


#operating system
df["OpSys"].value_counts()


# In[84]:


sns.barplot(df["OpSys"],df["Price"])
plt.xticks(rotation="vertical")


# In[85]:


def setcatagory(text):
    if text == "Windows 10" or text == "Windows 7" or text == "Windows 10 S":
        return "Windows"
    elif text == "Mac OS X" or text == "macOS":
        return "Mac"
    else:
        return "other"
df["OpSys"] = df["OpSys"].apply(lambda x:setcatagory(x))
       
    


# In[86]:


df["OpSys"].value_counts()


# In[87]:


df


# In[88]:


#weight analysis 
sns.distplot(df["Weight"])


# In[89]:


sns.scatterplot(df["Weight"],df["Price"])


# In[90]:


#Price Distribution
sns.distplot(df["Price"])


# In[91]:


#Applying log function
sns.distplot(np.log(df["Price"]))


# In[92]:


df


# In[93]:


df["Company"].value_counts()


# In[ ]:





# In[ ]:





# In[94]:


#coorelation
#correlation
plt.figure(figsize = (15,10))
sns.heatmap(data = df.corr(),cmap='plasma',annot=True,linewidths=0.5)
plt.show()


# In[95]:


df["Company"].value_counts()


# In[96]:


df["Gpu brand"].value_counts()


# In[97]:


df


# # model Building
# 

# In[98]:


dfs = pd.get_dummies(df)
dfs


# In[99]:


x = dfs.drop(["Price"],axis= 1)
y = np.log(df["Price"])


# In[100]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


# In[101]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=15)
x_train.shape,x_test.shape


# In[102]:


models = [
    ("LinearRegression",LinearRegression()),
    ("Lasso",Lasso(alpha=10)),
    ("Ridge",Ridge(alpha=0.001)),
    ("RandomForestRegressor", RandomForestRegressor(max_depth=15,max_features=0.75,max_samples=0.5,n_estimators=100)),
    ("DecisionTreeRegressor",DecisionTreeRegressor(max_depth=8)),
    ("KNeighborsRegressor",KNeighborsRegressor())
]


# In[103]:


for name,model in models:
    print(name)
    print()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("mean_squared_error:",mean_squared_error(y_test,y_pred))
    print("\n")
    print("r2_score:",r2_score(y_test,y_pred))
    print("\n")
    print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))
    print("\n")
    


# In[ ]:


#High performance model

#RandomForestRegressor

#mean_squared_error: 0.043926072361411404


#r2_score: 0.875713580392166


#mean_absolute_error: 0.16375274820092994


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # MODEL BUILDING

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #LinearRegression

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




