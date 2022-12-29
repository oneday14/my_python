# 딥러닝을 이용한 자연어 처리 입문
# https://wikidocs.net/book/2155
# 02-02까지 작업

## 용어 
# - 구두점(punctuation) : 마침표(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 등과 같은 기호
# - 띄어쓰기(whitespace)
# - 접어(clitic) : 단어가 줄임말로 쓰일 때 생기는 형태 (ex. 'm, 're)
# - 구분자(boundary) : 문장의 구분 기호
# - 어절 : 한국어에서 띄어쓰기 단위가 되는 단위
# - 교착어 : 조사, 어미 등을 붙여서 말을 만드는 언어 (ex. 한국어)
# - 형태소(morpheme) : 뜻을 가진 가장 작은 말의 단위
# --- 자립 형태소 : 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소. 그 자체로 단어가 됨. 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등을 말함.
# ----- 어간(stem) : 단어의 의미를 담고 있는 단어의 핵심 부분
# --- 의존 형태소 : 다른 형태소와 결합하여 사용되는 형태소. 접사, 어미, 조사, 어간을 말함.
# ----- 접사(affix) : 단어에 추가적인 의미를 주는 부분
# - 표제어(lemma) : 기본 사전형 단어

## 토큰화 
# - 토큰화(Tokenization) : 주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업
# --- 단어 토큰화(Word Tokenization) : 토큰의 기준을 단어(word)로 하는 경우. 단, 단어(word)는 단어 단위 외에도 단어구, 의미를 갖는 문자열로도 간주되기도 함
# --- 문장 토큰화(Sentence Tokenization) : 토큰의 단위가 문장(sentence)일 경우 (= 문장 분류(sentence segmentation))

# - 고려사항
# --- 구두점이나 특수문자를 전부 제거하면 토큰이 의미를 잃어버리는 경우가 발생
# --- 한국어는 띄어쓰기만으로는 단어 토큰을 구분하기 어려움
# --- 토큰화 도구 선택은 해당 데이터를 가지고 어떤 용도로 사용할 것인지에 따라서 그 용도에 영향이 없는 기준으로 정하면 됨

# - 영어 단어 토큰화 도구 종류
# --- word_tokenize
# ----- 축약형 분리
from nltk.tokenize import word_tokenize
print('단어 토큰화1 :',word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# =============================================================================
# 단어 토큰화1 : ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
# =============================================================================

# --- WordPunctTokenizer
# ----- 구두점으로 분리
from nltk.tokenize import WordPunctTokenizer
print('단어 토큰화2 :',WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# =============================================================================
# 단어 토큰화2 : ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']  
# =============================================================================

# --- keras의 text_to_word_sequence
# ----- 축약형 미분리, 구두점 제거, 소문자 변경
from tensorflow.keras.preprocessing.text import text_to_word_sequence
print('단어 토큰화3 :',text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# =============================================================================
# 단어 토큰화3 : ["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
# =============================================================================

# --- Penn Treebank Tokenization
# ----- 하이픈('-') 한단어로 유지, 축약형 분리
from nltk.tokenize import TreebankWordTokenizer
print('단어 토큰화4 :',TreebankWordTokenizer().tokenize("Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."))
# =============================================================================
# 단어 토큰화4 : ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
# =============================================================================

# - 한국어 단어 토큰화 도구 종류
# --- OKT
from konlpy.tag import Okt
okt = Okt()
print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
# =============================================================================

# --- Kkma (꼬꼬마)
from konlpy.tag import Kkma
kkma = Kkma()
print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 꼬꼬마 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
# =============================================================================

# --- Komoran (코모란)
from konlpy.tag import Komoran
komoran = Komoran() 
print('코모란 형태소 분석 :',komoran.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 코모란 형태소 분석 : ['열심히', '코', '딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가', '아', '보', '아요']
# =============================================================================

# --- Hannanum (한나눔)
from konlpy.tag import Hannanum
hannanum = Hannanum()
print('한나눔 형태소 분석 :',hannanum.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 한나눔 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에는', '여행', '을', '가', '아', '보', '아']
# =============================================================================

# --- Mecab (메캅)
# ----- 윈도우 지원 X
! git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
! bash Mecab-ko-for-Google-Colab/install_mecab-ko_on_colab190912.sh
from eunjeon import Mecab   
mecab = Mecab() 
print('메캅 형태소 분석 :',mecab.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 메캅 형태소 분석 : 
# =============================================================================

# - 문장 토큰화 도구 종류
# --- sent_tokenize
from nltk.tokenize import sent_tokenize
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화1 :',sent_tokenize(text))
# =============================================================================
# 문장 토큰화1 : ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
# =============================================================================  

# --- KSS(Korean Sentence Splitter)
# ----- 한국어 문장 토큰화 도구
!pip install kss
import kss
text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))
# =============================================================================
# 한국어 문장 토큰화 : ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', '이제 해보면 알걸요?']
# =============================================================================

## 태깅
# - 품사 태깅 (Part-of-speech tagging) : 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지 구분하는 작업

# - 영어 품사 태깅 종류
# --- Penn Treebank POS Tags
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))

# =============================================================================
# 단어 토큰화 : ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
# 품사 태깅 : [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
# =============================================================================

# - 한국어 품사 태깅 종류
# --- OKT
from konlpy.tag import Okt
okt = Okt()
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
# =============================================================================

# --- Kkma (꼬꼬마)
from konlpy.tag import Kkma
kkma = Kkma()
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 꼬꼬마 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
# =============================================================================

# --- Komoran (코모란)
from konlpy.tag import Komoran
komoran = Komoran() 
print('코모란 품사 태깅 :',komoran.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 코모란 품사 태깅 : [('열심히', 'MAG'), ('코', 'NNG'), ('딩', 'MAG'), ('하', 'XSV'), ('ㄴ', 'ETM'), ('당신', 'NNP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKB'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가', 'VV'), ('아', 'EC'), ('보', 'VX'), ('아요', 'EC')]
# =============================================================================

# --- Hannanum (한나눔)
from konlpy.tag import Hannanum
hannanum = Hannanum()
print('한나눔 품사 태깅 :',hannanum.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 한나눔 품사 태깅 : [('열심히', 'M'), ('코딩', 'N'), ('하', 'X'), ('ㄴ', 'E'), ('당신', 'N'), (',', 'S'), ('연휴', 'N'), ('에는', 'J'), ('여행', 'N'), ('을', 'J'), ('가', 'P'), ('아', 'E'), ('보', 'P'), ('아', 'E')]
# =============================================================================

# --- Mecab (메캅)
# ----- 윈도우 지원 X
! git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
! bash Mecab-ko-for-Google-Colab/install_mecab-ko_on_colab190912.sh
from eunjeon import Mecab   
mecab = Mecab() 
print('메캅 품사 태깅 :',mecab.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# =============================================================================
# 메캅 품사 태깅 : 
# =============================================================================

## 정제
# - 정제(cleaning) : 코퍼스로부터 노이즈 데이터를 제거

# - 고려사항
# --- 정제 작업은 토큰화 작업에 방해가 되는 부분들을 배제시키고 토큰화 작업을 수행하기 위해서 토큰화 작업보다 앞서 이루어지기도 하지만, 토큰화 작업 이후에도 여전히 남아있는 노이즈를 제거하기위해 지속적으로 이루어지기도 함
# --- 완벽한 정제 작업은 어려운 편, 대부분의 경우 이 정도면 됐다.라는 일종의 합의점을 찾음

# - 정제 방법 종류
# --- 등장 빈도가 적은 단어 제외
# --- 길이가 짧은 단어 제외







## 정규화
# - 정규화(normalization) : 표현 방법이 다른 단어를 통합시켜서 같은 단어로 만들어줌. 코퍼스로부터 복잡성을 줄이는 작업.
    
# - 주의사항
# --- 대, 소문자 통합 방법이 있으나, 무작정 통합해서는 안 됨. 미국을 뜻하는 단어 US와 우리를 뜻하는 us처럼 구분해야하는 상황 존재

# - 정규화 방법 종류
# --- 어간 추출(stemming) : 어간(Stem)을 추출하는 작업. 품사 정보가 보존되지 않음. 어간 추출을 한 결과는 사전에 존재하지 않는 단어일 경우가 많음.
# --- 표제어 추출(lemmatizaiton) : 단어들로부터 표제어를 찾아가는 과정. 표제어 추출기(lemmatizer)가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있음. 문맥을 고려하며 수행했을 때의 결과는 해당 단어의 품사 정보를 보존.
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('표제어 추출 후 :',[lemmatizer.lemmatize(word) for word in words])
# =============================================================================
# 표제어 추출 후 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
# =============================================================================
print('품사 정보 입력 후, 표제어 추출 :',lemmatizer.lemmatize('dies', 'v'))
# =============================================================================
# 품사 정보 입력 후, 표제어 추출 : 'die'
# =============================================================================

## 불용어 (= 노이즈 데이터(noise data))
