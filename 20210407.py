from selenium import webdriver
from bs4 import BeautifulSoup

# 웹페이지 접속
browser =  webdriver.Chrome('C:/informs/chromedriver.exe')

lst_num = []
lst_sym = []

for i in range(0, 461, 20) :
    url = 'https://okky.kr/articles/community?query=%EA%B0%95%EC%9D%98&offset='+ str(i) +'&max=20&sort=id&order=desc'
    browser.get(url)
    browser.implicitly_wait(5)      # 최대 5초 대기

    # 소스 읽어오기
    html = browser.page_source
    soup = BeautifulSoup(html, 'lxml')

    num = soup.select('span.list-group-item-text.article-id')
    sym = soup.select('div.list-group-item-summary.clearfix > ul > li')
    
    # 좋아요 수만 추출
    for j in range(0, len(sym)) :
        if j % 3 == 1:
            lst_sym.append(sym[j].text.replace(' ','').replace('\n', ''))
        else :
            pass
    # 글 숫자 추출             
    for z in num :
        lst_num.append(z.text)
    
lst_sym[0:20]
lst_num

# 데이터 프레임으로 만들기
import pandas as pd
df = pd.DataFrame({'number': lst_num, 'thumb': lst_sym})

from pandas import *

# 추천 수가 음수가 아닌 경우만 추출(음수일 경우 광고일 가능성이 크기때문)
df = df[df.thumb >= 0]
df.number = Series(list(map(lambda x : x.replace('#',''), df.number)))

# 각 페이지의 내용 긁어오기
lst_con = []
for i in df.number :
    url = 'https://okky.kr/article/'+ str(i)
    browser.get(url)
    browser.implicitly_wait(5)     

    # 소스 읽어오기
    html = browser.page_source
    soup = BeautifulSoup(html, 'lxml')

    con = soup.select('article.content-text > p')
    
    content = ''
    for j in con :
        content += j.text

    lst_con.append(content)
    
# 새로운 칼럼 추가
df['content'] = Series(lst_con)

# 이모티콘 및 특수기호 제거
import re
r1 = re.compile('[a-zA-Z가-힣0-9]+[a-zA-Z가-힣0-9]', flags = re.IGNORECASE)

# 순수한 내용 칼럼 추가
df['pure_content'] = Series([' '.join(i) for i in df.content.str.findall(r1)])

# csv파일 
df.to_csv('okky크롤링.csv', index = False, encoding='utf-8-sig')
