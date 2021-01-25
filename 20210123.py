from pandas import *
import pandas as pd

dt = pd.read_csv('테스트 로데이터.csv')

con = dt.loc[:,'Content']
ttl = dt.loc[:,'Title']

con = con.astype('str')
ttl = ttl.astype('str')

# 정규식 표현
import re
r1 = re.compile("[a-zA-Z']+[a-zA-Z]", flags = re.IGNORECASE)

con_com = Series(con).str.findall(r1)
ttl_com = Series(ttl).str.findall(r1)

dt = con_com + ttl_com

# 불용어 처리
import nltk                           
from nltk.corpus import stopwords     # 불용어추가
nltk.download('stopwords')            # 불용어사전

sw = nltk.corpus.stopwords.words('english')
newstopwords = ['would', 'could', 'the', 'is', 'it', 'not', 'was']
sw.extend(newstopwords)

for i in dt :                         # 불용어 삭제
    for j in i :
        if j.lower() in sw :
            i.remove(j)

# 연관분석
# pip install --no-binary :all: mlxtend # mlxtend 설치

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dt_lst = list(dt)

te = TransactionEncoder()
transaction = te.fit(dt_lst).transform(dt_lst)
te.columns_

df = DataFrame(transaction, columns = te.columns_)

# 지지도
#  - 전체 거래 중 A와 B를 동시에 구매한 확률
# 신뢰도
#  - A를 구매했을때 B도 구매한 확률
# 향상도
#  - 신뢰도/P(B)
#  - 두 아이템의 연관 규칙이 우연인지 아닌지를 나타내는 척도
#  - 향상도가 1이면 독립, 1보다 크면 양의 상관관계, 1보다 작으면 음의 상관관계

f1 = apriori(df, min_support = 0.1, use_colnames=True)         # min_support : 최소지지도, 디폴트 값은 0.5
# =============================================================================
#      support         itemsets
# 0   0.149158         (Kimchi)
# 1   0.109463        (cabbage)
# 2   0.129511         (flavor)
# 3   0.251002           (good)
# 4   0.403769         (kimchi)
# 5   0.214515           (like)
# 6   0.121492        (product)
# 7   0.152767          (spicy)
# 8   0.206496          (taste)
# 9   0.122694   (good, kimchi)
# 10  0.128308   (kimchi, like)
# 11  0.113071  (kimchi, taste)
# =============================================================================

from mlxtend.frequent_patterns import association_rules

association_rules(f1, metric='confidence', min_threshold=0.1)  # min_threshold : 최소신뢰도

# =============================================================================
#   antecedents consequents  antecedent support  ...      lift  leverage  conviction
# 0      (good)    (kimchi)            0.251002  ...  1.210637  0.021347    1.166377
# 1    (kimchi)      (good)            0.403769  ...  1.210637  0.021347    1.075949
# 2    (kimchi)      (like)            0.403769  ...  1.481369  0.041693    1.151359
# 3      (like)    (kimchi)            0.214515  ...  1.481369  0.041693    1.483644
# 4    (kimchi)     (taste)            0.403769  ...  1.356154  0.029695    1.102150
# 5     (taste)    (kimchi)            0.206496  ...  1.356154  0.029695    1.317850
# =============================================================================

# 빈도 확인
lst = []
cnt = []
for i in dt :
    for j in i :
        if j.lower() in lst:
            cnt[lst.index(j.lower())] += 1 
        else :
            lst.append(j.lower())
            cnt.append(1)
                
df2 = DataFrame([lst, cnt]).T
df2.columns = ['word', 'cnt']                
df2 = df2.sort_values('cnt', ascending = False)

df2.iloc[:10,:]
# =============================================================================
#           word   cnt
# 13      kimchi  2664
# 12        good  1016
# 152      taste   762
# 42        like   747
# 116      spicy   541
# 134      great   483
# 32      flavor   412
# 50     product   412
# 109  delicious   404
# 69         one   390
# =============================================================================
