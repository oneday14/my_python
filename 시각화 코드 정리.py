import seaborn as sns
sns.distplot(data, color=color)  # 정규분포
plt.axvline(sum(data)/len(data), color=color, linestyle='--') # 수직선 그리기

# 시각화 한글 font 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.title(title)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# bar 그래프 그리기
color = ['c','m','y']
plt.bar(x_lst, y_lst, color=color, label=label)
plt.ylim([y_min, y_max])
plt.xticks(x_lst,label_lst)
plt.title(title, fontsize=16)
plt.text(x, y, text)
plt.legend()

# pie 그래프 그리기
ratio = data
labels = label_lst
plt.pie(ratio, labels=labels, autopct='%.1f%%', colors=color) # autopct : 소수점 자리수
plt.title(title, fontsize = 16)

