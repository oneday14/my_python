# 라이브러리 불러오기
from selenium import webdriver as wd 
from bs4 import BeautifulSoup 
import time 

# 웹페이지 접속
driver = wd.Chrome('C:/informs/chromedriver.exe') 
url = 'https://www.youtube.com/watch?v=fE2h3lGlOsk' 
driver.get(url) 

# 하단으로 내리기
last_page_height = driver.execute_script("return document.documentElement.scrollHeight") 

# 소스 가져오기
while True: 
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);") 
    time.sleep(3.0) 
    new_page_height = driver.execute_script("return document.documentElement.scrollHeight") 
    
    if new_page_height == last_page_height: 
        break 
    last_page_height = new_page_height 
    
html_source = driver.page_source 

driver.close() 

soup = BeautifulSoup(html_source, 'lxml')

# 댓글 가져오기
youtube_comments = soup.select('yt-formatted-string#content-text')

youtube_comments[0].text

lst_comments = []

for i in youtube_comments:
    lst_comments.append(i.text)

