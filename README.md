# 文本分類實現 - Python (Gensim, Keras)

### 國立政治大學 111 學年度 人工智慧應用專題 『AI 主流應用及技巧』實作專案
自然語言處理包含：語意分析(Semantic Analysis)、命名實體(NER)、文本分類(Text Categorization)、翻譯(Translaton) …… 等任務，
這裡我基於 [【人工智慧應用專題】人工智慧主流技術 課程](https://www.youtube.com/watch?v=vchetga-8-M)，劉瓊如老師在第一次與第四次上課所介紹的語言模型(詞向量表示法)，利用 python 實作。
另外，初學ML和NLP，主要根據老師上課、炎龍老師YT頻道還有網路上東看看西看看自學而來，如果有錯誤再請多多指教

### Data
資料是 kaggle Dou ban Movie short comments 豆瓣電影評論
- [資料連結 Kaggle](https://www.kaggle.com/datasets/liujt14/dou-ban-movie-short-comments-10377movies)
- [資料連結 my google drive](https://drive.google.com/file/d/1K4GNFKhjJEsBOTJ0riZaSEAXPfkZy6Vp/view?usp=share_link)
-----
## 資料預處理
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
```
```py
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
## 轉換詞向量
- tfidf
```py
ngram_range=(1, 2)
max_feature = 1000
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        max_features=max_feature,
                        norm='l2',)
features_train_tfdif = tfidf.fit_transform(comment_train)
features_test_tfidf = tfidf.transform(comment_test)
```
  - 一樣可以利用 tfidf 去看正負評關鍵字
```py
tfdif_score_all = pd.DataFrame({'全部評論': tfidf.get_feature_names(),
                   'tfidf_score': features_train_tfdif.toarray().mean(axis=0).tolist()}).sort_values(by='tfidf_score', ascending=False).reset_index(drop=True)
tfdif_score_pos = pd.DataFrame({'正面評論': tfidf.get_feature_names(),
                   'tfidf_score': features_train_tfdif.toarray()[y_train==1].mean(axis=0).tolist()}).sort_values(by='tfidf_score', ascending=False).reset_index(drop=True)
tfdif_score_neg = pd.DataFrame({'負面評論': tfidf.get_feature_names(),
                   'tfidf_score': features_train_tfdif.toarray()[y_train==0].mean(axis=0).tolist()}).sort_values(by='tfidf_score', ascending=False).reset_index(drop=True)
pdWordTfidf = pd.concat([tfdif_score_all, tfdif_score_pos, tfdif_score_neg], axis=1)
del tfdif_score_all, tfdif_score_pos, tfdif_score_neg
```

### Word2Vec
```py
from gensim.models import word2vec
import logging
import multiprocessing
import random
logging.basicConfig(format = '%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
```
  - word2vec, doc2vec 需要資料格式為 list of list
```py
# comment lists to list of lists with tokens
comment_train_list_of_list = [i.split(' ') for i in comment_train]
comment_test_list_of_list = [i.split(' ') for i in comment_test]
```
  - train
```py
w2v = word2vec.Word2Vec(size=128, min_count=5, window=5, sg=1)
w2v.build_vocab(comment_train_list_of_list)
# training
for i in range(20):
  random.shuffle(comment_train_list_of_list)
  w2v.train(comment_train_list_of_list, total_examples=len(comment_train_list_of_list), epochs=1)
```

```py
# get vectors
w2v.wv['好看
# get similar words
w2v.wv.most_similar('好看')
def most_similar_w2v(words, topn=10):
  similar_df = pd.DataFrame()
  for word in words:
    try:
      similar_words = pd.DataFrame(w2v.wv.most_similar(word, topn=topn), columns=[word, 'cos similarity'])
      similar_df = pd.concat([similar_df, similar_words], axis=1)
    except:
      print(word, 'not found in Word2Vec model')
  return similar_df
 most_similar_w2v(words=['好看', 'marvel', '超英', '英雄', '感人', '動畫', 'XD'])
```
- 將文章轉為詞向量 為之後套入 xgb 準備
```py
# 先篩選掉模型沒看過的字
comment_test_list_of_list_filtered = [[word for word in com if  word in w2v.wv] for com in comment_test_list_of_list]
comment_train_list_of_list_filtered = [[word for word in com if  word in w2v.wv] for com in comment_train_list_of_list]

features_train_w2v  = []
features_test_w2v = []

for com in comment_train_list_of_list_filtered:
  if len(com) == 0:
    features_train_w2v.append(np.array([0] * 128))
  else:
    features_train_w2v.append(np.mean([w2v.wv[w] for w in com], axis=0))

for com in comment_test_list_of_list_filtered:
  if len(com) == 0:
    features_test_w2v.append(np.array([0] * 128))
  else:
    features_test_w2v.append(np.mean([w2v.wv[w] for w in com], axis=0))
 
features_test_w2v = np.stack(features_test_w2v)
features_train_w2v = np.stack(features_train_w2v)

```
### Doc2Vec
```py
from gensim.models import Doc2Vec, doc2vec
d2v = Doc2Vec(size=128,
              min_count=5,
              window=5)
comment_tag_list = []
for ind, com in enumerate(comment_train_list_of_list):
  comment_tag_list.append(doc2vec.LabeledSentence(words=com, tags=[str(ind)]))
```

```py
comment_tag_list[:2]
[LabeledSentence(words=['晚上', '陪欣姐', '中老年', '愛情劇', '女兒', '單身', '爸爸', '喜歡', '一個', '開心', '人開', '開心', '愛呢', '努力', '開心', '開心', '大概', '劉鶯鶯', '喜歡', '放肆'], tags=['0']),
 LabeledSentence(words=['畢竟', '原著', '震撼', '至少', '一部', '打哈欠', '看不下去', '片子'], tags=['1'])]
```
- train
```py
d2v.build_vocab(comment_tag_list)
epoch = 20
for i in range(epoch):
  print(f'epoch: {i+1}/{epoch}')
  random.shuffle(comment_tag_list)
  d2v.train(comment_tag_list, total_examples=len(comment_tag_list), epochs=1)
d2v.save('preTrainModel/d2v')
```

```py
d2v.most_similar('好看', topn=10)
[('精彩', 0.6138744354248047),
 ('喜歡', 0.575271487236023),
 ('不錯', 0.5678427219390869),
 ('有意思', 0.5677773356437683),
 ('感人', 0.5598394870758057),
 ('很不錯', 0.5501113533973694),
 ('還不錯', 0.539584755897522),
 ('挺不錯', 0.5301656723022461),
 ('實話', 0.5283814668655396),
 ('很棒', 0.5279253721237183)]
```

- 計算單詞相似度 與 新文章與訓練文章 topn 相似
```py
x = '好看很帥緊湊，超喜歡的，不懂為何有人分數給那麼低,真的不錯阿'
x = CutContent(x).split(' ')
print(x)
['好看', '很帥', '超喜歡', '為何', '有人', '分數給', '那麼', '真的', '不錯']
###################
test_d2v = d2v.infer_vector(x, alpha=0.025, steps=300)
sims = d2v.docvecs.most_similar([test_d2v], topn=10)

for count, sim in sims:
  print([sen for sen in comment_tag_list if sen.tags==[str(count)]])
  print(sim)
 [LabeledSentence(words=['真心', '有人', '值星'], tags=['53715'])]
0.6500393152236938
[LabeledSentence(words=['有人', '真心', '還不錯', '記住', '記住'], tags=['212970'])]
0.625268280506134
[LabeledSentence(words=['真的', '有人', '好看', '真的', '有淚點', '期望值', '太高'], tags=['257919'])]
0.6123305559158325
[LabeledSentence(words=['有人', '特別', '喜歡', '天太逗', '最喜', '歡蠢'], tags=['340803'])]
0.5940371751785278
[LabeledSentence(words=['真心', '有人', '好看', '亮點'], tags=['250748'])]
0.5912108421325684
[LabeledSentence(words=['改編', '顛覆', '原著', '有人', '說壞', '很帥', '暴走', '大蟲子', '噴飯', '私定', '終身'], tags=['184445'])]
0.5881091356277466
[LabeledSentence(words=['還不錯', '有人', '不愛錢'], tags=['207050'])]
0.5781245231628418
[LabeledSentence(words=['好看', '開場', '兩分鐘', '有人', '唱歌跳舞', '電影'], tags=['299283'])]
0.564518928527832
[LabeledSentence(words=['講真', '真的', '有人'], tags=['310733'])]
0.5623056292533875
[LabeledSentence(words=['真的', '好看', '真的', '有人', '贏了'], tags=['129929'])]
0.560077428817749
```
- 轉成詞向量
因為 gendim dev2vec ``infer_vector`` 可以設定 alpha(learning rate)、steps(epochs)讓新文章更好的轉換向量，但速度並不快，這邊我使用平行運算加速。

```py
from multiprocessing import Pool
d2v.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)
def infer_vector_worker(comment):
  vec = d2v.infer_vector(comment, alpha=0.025)
  return vec
 
train['list_of_list_Comment'] = comment_train_list_of_list
test['list_of_list_Comment'] = comment_test_list_of_list
features_train_d2v = train['list_of_list_Comment'].apply(lambda x: d2v.infer_vector(x, alpha=0.025))
features_test_d2v = test['list_of_list_Comment'].apply(lambda x: d2v.infer_vector(x, alpha=0.025))
features_train_d2v = np.stack(features_train_d2v.values)
features_test_d2v = np.stack(features_test_d2v.values)
```

## 套入分類器
因為是簡單實現文本分類，這邊使用 XGBoost
```py
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
```
- tfidf
```py
xgb = XGBClassifier()
xgb.fit(features_train_tfdif, y_train)
print(' ---- confusion matrix ---------')
print(confusion_matrix(y_test, xgb.predict(features_test_tfidf)))
print(' ---- classification report ----------')
print(classification_report(y_test, xgb.predict(features_test_tfidf)))

---- confusion matrix ---------
[[10523 49684]
 [ 2769 88081]]
 ---- classification report ----------
              precision    recall  f1-score   support

           0       0.79      0.17      0.29     60207
           1       0.64      0.97      0.77     90850

    accuracy                           0.65    151057
   macro avg       0.72      0.57      0.53    151057
weighted avg       0.70      0.65      0.58    151057
```
- word2vec
```py
xgb = XGBClassifier()
xgb.fit(features_train_w2v, y_train)
pred = xgb.predict(features_test_w2v)
print(' ---- confusion matrix ---------')
print(confusion_matrix(y_test, pred))
print(' ---- classification report ----------')
print(classification_report(y_test, pred))

 ---- confusion matrix ---------
[[   20 60187]
 [   41 90809]]
 ---- classification report ----------
              precision    recall  f1-score   support

           0       0.33      0.00      0.00     60207
           1       0.60      1.00      0.75     90850

    accuracy                           0.60    151057
   macro avg       0.46      0.50      0.38    151057
weighted avg       0.49      0.60      0.45    151057
```

- doc2vec
```py
xgb = XGBClassifier()
xgb.fit(features_train_d2v, y_train)
pred = xgb.predict(features_test_d2v)
print(' ---- confusion matrix ---------')
print(confusion_matrix(y_test, pred))
print(' ---- classification report ----------')
print(classification_report(y_test, pred))

 ---- confusion matrix ---------
[[   77 60130]
 [  114 90736]]
 ---- classification report ----------
              precision    recall  f1-score   support

           0       0.40      0.00      0.00     60207
           1       0.60      1.00      0.75     90850

    accuracy                           0.60    151057
   macro avg       0.50      0.50      0.38    151057
weighted avg       0.52      0.60      0.45    151057
```

## LSTM implement
```py
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
```

```py
tokenizer = Tokenizer(num_words=50000) # 設定最常使用的 50,000 字
tokenizer.fit_on_texts(comment_train)
x_train = tokenizer.texts_to_sequences(comment_train)
x_test = tokenizer.texts_to_sequences(comment_test)

x_train = sequence.pad_sequences(x_train, maxlen=128)
x_test = sequence.pad_sequences(x_test, maxlen=128)

lstm_model = Sequential()
lstm_model.add(Embedding(50000, 128))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(10, activation='relu'))
lstm_model.add(Dense(1, activation='sigmoid'))
```

```py
lstm_model.summary()

Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, None, 128)         6400000   
                                                                 
 lstm_2 (LSTM)               (None, 128)               131584    
                                                                 
 dense_3 (Dense)             (None, 10)                1290      
                                                                 
 dense_4 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 6,532,885
Trainable params: 6,532,885
Non-trainable params: 0
_________________________________________________________________
```

```py
lstm_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
lstm_model.fit(x_train, y_train,
               batch_size=512,
               epochs=5,
               validation_data=[x_test, y_test])
Epoch 1/5
689/689 [==============================] - 1296s 2s/step - loss: 0.5150 - accuracy: 0.7407 - val_loss: 0.4852 - val_accuracy: 0.7635
Epoch 2/5
689/689 [==============================] - 1252s 2s/step - loss: 0.4471 - accuracy: 0.7851 - val_loss: 0.4869 - val_accuracy: 0.7642
Epoch 3/5
689/689 [==============================] - 1239s 2s/step - loss: 0.4045 - accuracy: 0.8059 - val_loss: 0.5147 - val_accuracy: 0.7585
Epoch 4/5
689/689 [==============================] - 1234s 2s/step - loss: 0.3603 - accuracy: 0.8275 - val_loss: 0.5733 - val_accuracy: 0.7544
Epoch 5/5
689/689 [==============================] - 1235s 2s/step - loss: 0.3179 - accuracy: 0.8490 - val_loss: 0.6508 - val_accuracy: 0.7463
```
