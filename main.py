import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#%%
df = pd.read_csv('allnames.tsv', delimiter='\t')


#%%

train = df[df['Train/Test'] == 'Train']
test = df[df['Train/Test'] == 'Test']


#%%

len(train)

#%%

from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams, everygrams

tokenizer = RegexpTokenizer(r'\w+')


def customer_analyzer(text):
    tokenized = tokenizer.tokenize(text.lower())
    # ng = ngrams((tokenized[0]), )
    ng = [''.join(x) for x in everygrams('#' + tokenized[0] + "#", 3, 11)]
    return ng + tokenized


# tmp = [ ''.join(x) for x in everygrams('daniel', 2, 4)]
#%%


# cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
# cv = CountVectorizer(token_pattern=r'\w+', lowercase=True)
cv = CountVectorizer(analyzer=customer_analyzer)
# cv = CountVectorizer(token_pattern=r'\w+', ngram_range=(2, 10), analyzer='char_wb', lowercase=True)
X_train_cv = cv.fit_transform(train['Person Name'])
X_test_cv = cv.transform(test['Person Name'])

#%%
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)

#%%

y_train = train['Gender'].apply(lambda x: 0 if x == 'Male' else 1)
y_test = test['Gender'].apply(lambda x: 0 if x == 'Male' else 1)

#%%

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

#%%

from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))

#%%