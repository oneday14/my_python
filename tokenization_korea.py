from konlpy.tag import Okt
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from konlpy.tag import Hannanum
from eunjeon import Mecab    # 메캅 (윈도우 지원 X)
# ! git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# ! bash Mecab-ko-for-Google-Colab/install_mecab-ko_on_colab190912.sh

okt = Okt()
kkma = Kkma()
komoran = Komoran() 
hannanum = Hannanum()
mecab = Mecab()  

text = '산뜻한 과일로 만든, 촉촉한 핸드크림'
print('OKT 형태소 분석 :',okt.morphs(text))
print('OKT 품사 태깅 :',okt.pos(text))
print('OKT 명사 추출 :',okt.nouns(text)) 

print('꼬꼬마 형태소 분석 :',kkma.morphs(text))
print('꼬꼬마 품사 태깅 :',kkma.pos(text))
print('꼬꼬마 명사 추출 :',kkma.nouns(text))  

print('코모란 형태소 분석 :',komoran.morphs(text))
print('코모란 품사 태깅 :',komoran.pos(text))
print('코모란 명사 추출 :',komoran.nouns(text))  

print('한나눔 형태소 분석 :',hannanum.morphs(text))
print('한나눔 품사 태깅 :',hannanum.pos(text))
print('한나눔 명사 추출 :',hannanum.nouns(text))  

print('메캅 형태소 분석 :',mecab.morphs(text))
print('메캅 품사 태깅 :',mecab.pos(text))  
print('메캅 명사 추출 :',mecab.nouns(text))  
