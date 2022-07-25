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
# 
