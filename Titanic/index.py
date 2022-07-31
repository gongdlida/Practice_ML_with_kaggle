import pandas as pd #pandas 라이브러리에 대해서 간단하게 정리
import matplotlib.pyplot as plt
import seaborn as sns

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

train.shape # train의 로우와 컬럼 수 891,12 
train.info() # Age / Cabin / Embarked 의 항목이 전체 행의 개수와 맞지 않음 -> 데이터 가공이 필요한 값이 있음
train.isnull().sum() # 데이터에서 null인 값들을 더한 후 보여줌
sns.set() # setting seaborn default for plots

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show()


bar_chart('Sex') # 성비율에 따른 생존률  남성 < 여성
bar_chart("Pclass") # 비행기 등급에 따른 생존률 3 < 2 < 1

# 총 승객수에서 가족과 함께 탄 사람들의 정확한 비율이 있어야 데이터를 해석할 수 있을 것 같음
bar_chart("SibSp") # 형제, 자매와 함께 탑승했을 경우의 생존률
bar_chart("Parch") # 가족과 함께 탑승했을 경우의 생존률

bar_chart("Embarked") # 선착장에 따른 생존률

# Feature Engineering 
  # feature -> column
  # Text를 컴퓨터가 이해할 수 있는 숫자로 변환해줘야 한다. 
  # outlier or NaN 처리

def countMissingValue(data,feature):
  target = data[feature].isnull().sum()
  print(target)

# def convertStrToNum(feature,data,mappingData):
#   for dataset in data:
#     dataset[feature] = dataset[feature].map(mappingData)
#   print(data[feature])

# Name
train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

 # train 테이블에서 이름 제거, 이때 이름에서 미혼,기혼 정보를 일 수 있기 때문에 이름을 따로 분류
train["Title"].value_counts()

# feature scaling
# 승객의 이름 중 미혼, 기혼을 알 수 있는 정보를 숫자로 변형 후 테이블에 저장
# 0:남성 1: 미혼여성 2: 기혼여성 3: 그외 이름
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
"Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
"Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
  dataset['Title'] = dataset['Title'].map(title_mapping)
train["Title"].value_counts()
#convertStrToNum("Title",train_test_data,title_mapping)
train.head()

# 이름을 이용해 나타낸 생존률
bar_chart("Title")

# 이름 데이터 삭제
train.drop('Name', axis=1, inplace=True) 
test.drop('Name', axis=1, inplace=True)

# Sex
countMissingValue(train,"Sex")

sex_mapping = {"male":0, "female":1}

for dataset in train_test_data:
  dataset["Sex"] = dataset["Sex"].map(sex_mapping)
train.head()
bar_chart("Sex")

# Age

countMissingValue(train,"Age")
# 나이는 10대,20대 처럼 군집분류
# missing value는 이름을 통해 분류한 성별의 평균으로 값을 채운다.
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True) 
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train.groupby("Title")["Age"].transform("median")
# missing value 개수 0 확인
countMissingValue(train,"Age")

# 나이에 따른 생존율 차트 확인
# 전체
# 차트를 통해서 알 수 있는 것은 20~30대가 가장 많이 생존한 것을 확인할 수 있다.
facet = sns.FacetGrid(train, hue="Survived",aspect=4) 
facet.map(sns.kdeplot,'Age',shade= True) 
facet.set(xlim=(0, train['Age'].max())) 
facet.add_legend()
plt.show()


# 0~20대
facet = sns.FacetGrid(train, hue="Survived",aspect=4) 
facet.map(sns.kdeplot,'Age',shade= True) 
facet.set(xlim=(0, train['Age'].max())) 
facet.add_legend()
plt.xlim(0, 20)
plt.show()

# 30~40대
facet = sns.FacetGrid(train, hue="Survived",aspect=4) 
facet.map(sns.kdeplot,'Age',shade= True) 
facet.set(xlim=(0, train['Age'].max())) 
facet.add_legend()
plt.xlim(30, 40)
plt.show()

# 40~60대
facet = sns.FacetGrid(train, hue="Survived",aspect=4) 
facet.map(sns.kdeplot,'Age',shade= True) 
facet.set(xlim=(0, train['Age'].max())) 
facet.add_legend()
plt.xlim(40, 60)
plt.show()

# 60~80대
facet = sns.FacetGrid(train, hue="Survived",aspect=4) 
facet.map(sns.kdeplot,'Age',shade= True) 
facet.set(xlim=(0, train['Age'].max())) 
facet.add_legend()
plt.xlim(60, 80)
plt.show()

# 연령대별 그룹화
# child: 0 | young: 1 | adult: 2 | mid-age: 3 | senior: 4

for dataset in train_test_data: # 이렇게 반복문으로 돌리면 에러가 출력됌, 하나씩 순회해서 수정해야함
  dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
for dataset in train_test_data:  
  dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
for dataset in train_test_data:  
  dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
for dataset in train_test_data:  
  dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
for dataset in train_test_data:  
  dataset.loc[ dataset['Age'] > 62, 'Age'] = 4  

#Embarked
# Emarked의 missing value: 2명 => 가장 많이 탑승한 곳으로 모두 세팅
train.Pclass.value_counts()
train.Embarked.value_counts()

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts() 
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts() 
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class'] 
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.show() 

train.Embarked.value_counts()
train.info()
# 탑승객의 S 항구에서 
for dataset in train_test_data:
 dataset['Embarked'] = dataset['Embarked'].fillna('S')
 
embarked_mapping = {"S":0,"C":1,"Q":2}

for dataset in train_test_data:
  dataset.Embarked = dataset.Embarked.map(embarked_mapping)

# Fare
testMissingValue = test[test.Fare.isnull() == True] # Test 데이터에서 Fare와 Cabin이 missing value인 인원의 티켓 등급과 항구 정보를 통해 평균값 적용
test.Ticket
fareMedian = test[(test.Pclass == 3) & (test.Embarked == 0)&(test.Age == 3)].Fare.median()
testMissingValue.Fare = fareMedian

countMissingValue(train,"Fare")
#가격과 생존률 관계
def targetChart(target,fr=0,to=0):
 facet = sns.FacetGrid(train, h1ue="Survived",aspect=4) 
 facet.map(sns.kdeplot,target,shade= True) 
 facet.set(xlim=(0, train[target].max())) 
 facet.add_legend()
 if(to !=0):
  plt.xlim(fr,to)
  plt.show()
  return
 plt.show()


targetChart("Fare",0,20)
targetChart("Fare",0,30)
targetChart("Fare")

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0
for dataset in train_test_data:    
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
for dataset in train_test_data:
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
for dataset in train_test_data:    
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

#Cabin
# Cabin의 missing value는 Fare 평균을 이용해 추측
train.Cabin.value_counts()

# ??
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.show()

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

countMissingValue(train, "Cabin")
# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
countMissingValue(train, "Cabin")

# SibSp와 Parch 항목을 Family Size 항목으로 묶어 추가 -> 동승자가 있을 경우의 생존율
# 동승자가 없을 경우 1, 그외 2이상의 값을 갖는다.
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

features_drop = ['Ticket', 'SibSp', 'Parch'] # 불필요 데이터 제거
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
train_data = train.drop('Survived', axis=1)
target = train['Survived']
