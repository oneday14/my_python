from selenium import webdriver
from bs4 import BeautifulSoup

# 웹페이지 접속
url = input('url을 입력하세요 > ')
browser =  webdriver.Chrome('C:/informs/chromedriver.exe')
browser.get(url)
browser.implicitly_wait(5)      # 최대 3초 대기

# 소스 읽어오기
html = browser.page_source
print(html)
soup = BeautifulSoup(html, 'lxml')

# 가져오기
title_scr = soup.select('div.elementor-widget-container div.elementor-text-editor.elementor-clearfix')
browser.close()
exclusion = ['수강후기', '한 글자도 수정하지 않은 생생한 후기!', '수강생 인터뷰', 
             '수강생분들의 목소리를 직접 들어보세요!', '수강생 출신 기업']

# 순수 가져오기
title_lst = []

for i in title_scr :
    if i.text in exclusion :
        del title_scr[title_scr.index(i):]
        

for i in title_scr :
    if '모집중인 강의' in i : 
        start = title_scr.index(i)

        for j in range(start + 2, len(title_scr) ) :
            title_lst.append(title_scr[j].text.replace('\n', ''))

len(title_lst) 

