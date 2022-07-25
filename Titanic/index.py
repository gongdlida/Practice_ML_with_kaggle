import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

train = pd.read_csv("Titanic/input/train.csv")
test = pd.read_csv("Titanic/input/test.csv")

# Data meaning
  # Survived  | 0 = No, 1 = Yes
  # pclass    | Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd      
  # sibsp     | siblings or(and) spouses aboard the Titanic | 배우자 또는 형제,자매와 함께 탑승한 승객 
  # parch     | parents or(and) children aboard the Titanic | 부모 또는 자식과 함께 탑승 명수
  # ticket    | Ticket number
  # cabin     | Cabin number | 객실정보
  # embarked  | Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton | 탑승 항구 정보 

print(train.head(80))
train.info()
# train.isnull().sum()
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


bar_chart('Sex')

# sns.set() # setting seaborn default for plots
