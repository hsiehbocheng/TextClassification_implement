# TextClassification_implement

### 國立政治大學 111 學年度 人工智慧應用專題 『AI 主流應用及技巧』實作專案
自然語言處理包含：語意分析(Semantic Analysis)、命名實體(NER)、文本分類(Text Categorization)、翻譯(Translaton) …… 等任務，
這裡我基於 【人工智慧應用專題】人工智慧主流技術 課程，劉瓊如老師在第一次與第四次上課所介紹的語言模型(詞向量表示法)，利用 python 實作。
另外，初學ML和NLP，主要根據老師上課、炎龍老師YT頻道還有網路上東看看西看看自學而來，如果有錯誤再請多多指教

-----
# 資料預處理
- 資料是 kaggle Dou ban Movie short comments 豆瓣電影評論，包含用戶資訊、電影、評論、星星數、Like 數，這邊我只留我想要的欄位，Comment(評論)、Star(星星數)，並且設定 Star > 3 是正面評論，否則為負面評論，也就是在做二元分類的任務，只預測該評論為正面or負面，另外，因為資料量有兩百多萬筆，所以我只抽取1/4，一部分的資料做實做。

- 簡體轉繁體
```py
!pip install opencc
from opencc import OpenCC
cc = OpenCC('s2t') # 簡體字轉繁體字
dmsc['Comment'] = dmsc['Comment'].apply(cc.convert)
```

- 斷詞與停用詞刪除
```py
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = stopwords.words('chinese')
stopWords = [cc.convert(word) for word in stopWords]

print(stopWords[:10])
['一', '一下', '一些', '一切', '一則', '一天', '一定', '一方面', '一旦', '一時']
```

- 包裝成 function
```py
urlFilter = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-|_|:| \.\.\. )*\b')

def CutContent(comment, stopWords=stopWords):
  parsed = jieba.lcut(comment)
  parsed2 = list(filter(lambda text: text not in stopWords and len(text) > 1, parsed))
  parsed2 = ' '.join(parsed2)

  return parsed2
 
def clean_Comment(data: pd.DataFrame, comment_col_name='Comment', urlFilter=urlFilter, stopWords=stopWords):
  data['parsed_Comment'] = data['Comment']
  data['parsed_Comment'] = data['parsed_Comment'].str.lower()
  data['parsed_Comment'] = data['parsed_Comment'].str.replace(urlFilter, '')
  data['parsed_Comment'] = data['parsed_Comment'].str.replace(r'\n|\.|\u3000| ', '')
  data['parsed_Comment'] = data['parsed_Comment'].str.replace(r"\d+\.?\d*", "")

  data = data.dropna(axis=0, subset=['parsed_Comment']).reset_index(drop=True)
  parsed_Comment = data['parsed_Comment'].apply(CutContent)
  data['parsed_Comment'] = parsed_Comment
  data.drop_duplicates(inplace=True)
  data.reset_index(drop=True, inplace=True)

  return data
```
- 利用字頻率看正負評關鍵字
```py
from collections import Counter
tokens_all = []
tokens_positive = []
tokens_negative = []

for comment in comments:
  tokens = str(comment).split(' ') 
  tokens_all += tokens
for comment in positive_comments:
  tokens = str(comment).split(' ') 
  tokens_positive += tokens
for comment in negative_comments:
  tokens = str(comment).split(' ') 
  tokens_negative += tokens

key_words_all = pd.DataFrame(sorted(Counter(tokens_all).items(), key=lambda x:x[1], reverse=True)).iloc[:50, ]
key_words_pos = pd.DataFrame(sorted(Counter(tokens_positive).items(), key=lambda x:x[1], reverse=True)).iloc[:50, ]
key_words_neg = pd.DataFrame(sorted(Counter(tokens_negative).items(), key=lambda x:x[1], reverse=True)).iloc[:50, ]

pdWordFreq = pd.concat([key_words_all, key_words_pos, key_words_neg], axis=1)
pdWordFreq.columns = ['全部評論關鍵字', 'Freq', '正面評論關鍵字', 'Freq', '負面評論關鍵字', 'Freq']

del key_words_all, key_words_pos, key_words_neg, tokens_all, tokens_positive, tokens_negative
```

這裡可以發現正反兩面常出現的詞其實很難看得出差別，除了反面有一個尷尬以外，這裡可能是因為用 3 顆星去二分不夠嚴謹。
