from selenium import webdriver
from bs4 import BeautifulSoup

# 웹페이지 접속
browser =  webdriver.Chrome('C:/informs/chromedriver.exe')

lst_num = []
lst_sym = []

for i in range(0, 461, 20) :
    url = 'https://okky.kr/articles/community?query=%EA%B0%95%EC%9D%98&offset='+ str(i) +'&max=20&sort=id&order=desc'
    browser.get(url)
    browser.implicitly_wait(5)      # 최대 3초 대기

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

# csv파일로 저장
df.to_csv('okky크롤링.csv', index = False, encoding = 'cp949')
