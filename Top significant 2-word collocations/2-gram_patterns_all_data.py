import subprocess
import nltk
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

def install_package():
    repo_url = 'https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer.git'
    try:
        subprocess.run(f'pip install git+{repo_url}', check=True, shell=True)
        print("Package installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package: {e}")
install_package()

#%%
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
lemmatizer = FrenchLefffLemmatizer()

STOP_WORDS_FR = set(['0', '1', '2', '3', 'a', 'ah', 'ai', 'aime', 'aller', 'alors', 'ans', 'apres', 'après', 'as', 'au',
    'aussi', 'autre', 'autres', 'aux', 'avais', 'avait', 'avant', 'avec', 'avez', 'avoir', 'b', 'bah', 'bcp',
    'beaucoup', 'bien', 'bon', 'bonjour', 'bonne', 'bref', 'c', "c'est", "c'était", 'ca', 'ce', 'cela',
    'celle', 'celui', 'ces', 'cest', 'cet', 'cetait', 'cette', 'ceux', 'chaque', 'chez', 'co', 'comme',
    'comment', 'compte', 'contre', 'coup', 'cours', 'crois', 'cétait', 'c’est', 'd', 'dans', 'de', 'deja',
    'depuis', 'des', 'detre', 'deux', 'dire', 'dis', 'dit', 'dm', 'dois', 'doit', 'donc', 'du', 'déjà',
    'dêtre', 'e', 'eh', 'elle', 'elles', 'en', 'encore', 'entre', 'envie', 'es', 'est', 'estce', 'et', 'etais', 'etait',
    'etc', 'ete', 'etes', 'etre', 'eu', 'f', 'faire', 'fais', 'fait', 'faites', 'faut', 'fois', 'font', 'g',
    'genre', 'gens', 'grave', 'gros', 'gt', 'h', 'hein', 'https', 'i', 'il', 'ils', 'j', "j'ai", "j'aime",
    "j'avais", "j'me", "j'suis", "j'vais", 'jai', 'jaime', 'jamais', 'javais', 'je', 'jen', 'jme', 'jour',
    'journee', 'journée', 'jsp', 'jsuis', 'jte', 'juste', 'jvais', 'jveux', 'jetais', 'jétais', 'j’ai', 'k', 'l', 'la',
    'le', 'les', 'leur', 'leurs', 'lol', 'lui', 'là', 'm', 'ma', 'maintenant', 'mais', 'mal', 'mdr', 'mdrr',
    'mdrrr', 'mdrrrr', 'me', 'meme', 'merci', 'mes', 'met', 'mettre', 'mieux', 'mis', 'mm',
    'moi', 'moins', 'moment', 'mon', 'monde', 'mtn', 'même', 'n', 'na', 'nan', 'ne', 'nest', 'ni', 'nn',
    'non', 'nos', 'notre', 'nous', 'o', 'of', 'oh', 'ok', 'on', 'ont', 'ou', 'ouais', 'oui', 'où', 'p', 'par',
    'parce', 'parle', 'pas', 'passe', 'pcq', 'pense', 'personne', 'peu', 'peut', 'peutetre', 'peutêtre', 'peux',
    'plus', 'pour', 'pourquoi', 'pq', 'pr', 'prend', 'prendre', 'prends', 'pris', 'ptdr', 'ptdrrr',
    'q', 'qd', 'qu', "qu'il", "qu'on", 'quand', 'que', 'quel', 'quelle', 'quelque', 'quelques',
    'quelquun', 'qui', 'quil', 'quils', 'quoi', 'quon', 'r', 'rien', 'rt', 's', 'sa', 'sais', 'sait', 'sans',
    'se', 'sera', 'ses', 'sest', 'si', 'sil', 'soir', 'soit', 'son', 'sont', 'suis', 'super', 'sur', 't',
    'ta', 'tas', 'te', 'tellement', 'temps', 'tes', 'tete', 'the', 'tjrs', 'tjs', 'toi', 'ton', 'toujours',
    'tous', 'tout', 'toute', 'toutes', 'tres', 'trop', 'trouve', 'trouvé', 'très', 'tt', 'tu', 'u',
    'un', 'une', 'v', 'va', 'vais', 'vas', 'veut', 'veux', 'via', 'vie', 'viens', 'voila', 'voilà', 'voir',
    'vois', 'voit', 'vont', 'vos', 'votre', 'vous', 'vrai', 'vraiment', 'vs', 'vu', 'w', 'wsh', 'x', 'xd',
    'y', 'ya', 'z', 'à', 'ça', 'ça', 'étais', 'était', 'été', 'êtes', 'être', '–—', '-', ''])

def preprocess_text_french(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove everything followed by '@'
    text = re.sub(r"@(\S+)", "", text)
    # Remove punctuation and numbers, keep French-specific characters
    text = re.sub(r'[^\w\sàâçéèêëîïôûùüÿñæœ]', '', text)
    # Tokenize text
    tokens = word_tokenize(text, language='french')
    # Remove custom stopwords
    tokens = [word for word in tokens if word not in STOP_WORDS_FR]
    # Lemmatization using NLTK
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    return ' '.join(tokens)

#%%
df = pd.read_csv('abc.csv') # File name can be changed
non_abusive_df = df[df['pred_label'] == 'non-abusive']
abusive_df = df[df['pred_label'] == 'abusive']

df.loc[:, 'text'] = df['text'].apply(preprocess_text_french)

non_abusive_df = df[df['pred_label'] == 'non-abusive'].copy()
non_abusive_df.loc[:, 'text'] = non_abusive_df['text'].apply(preprocess_text_french)

abusive_df = df[df['pred_label'] == 'abusive'].copy()
abusive_df.loc[:, 'text'] = abusive_df['text'].apply(preprocess_text_french)

#%%
vectorizer = TfidfVectorizer(ngram_range=(2, 2))

#%%
tfidf_df = vectorizer.fit_transform(df['text'])
terms_df = vectorizer.get_feature_names_out()
tfidf_scores_df = tfidf_df.mean(axis=0).tolist()[0]
sorted_indices_df = tfidf_df.mean(axis=0).A.ravel().argsort()[::-1]
top_terms_df = [(terms_df[idx], tfidf_scores_df[idx]) for idx in sorted_indices_df[:100]]
print("Top significant 2-word collocations in all tweets:")
for idx, (term, score) in enumerate(top_terms_df, 1):
    print(f"{idx}. {term}: {score}")

#%%
tfidf_non_abusive = vectorizer.fit_transform(non_abusive_df['text'])
terms_non_abusive_df = vectorizer.get_feature_names_out()
tfidf_scores_non_abusive = tfidf_non_abusive.mean(axis=0).tolist()[0]
sorted_indices_non_abusive = tfidf_non_abusive.mean(axis=0).A.ravel().argsort()[::-1]
top_terms_non_abusive = [(terms_non_abusive_df[idx], tfidf_scores_non_abusive[idx]) for idx in sorted_indices_non_abusive[:100]]
print("Top significant 2-word collocations in non-abusive tweets:")
for idx, (term, score) in enumerate(top_terms_non_abusive, 1):
    print(f"{idx}. {term}: {score}")

#%%
tfidf_abusive = vectorizer.fit_transform(abusive_df['text'])
terms_abusive_df = vectorizer.get_feature_names_out()
tfidf_scores_abusive = tfidf_abusive.mean(axis=0).tolist()[0]
sorted_indices_abusive = tfidf_abusive.mean(axis=0).A.ravel().argsort()[::-1]
top_terms_abusive = [(terms_abusive_df[idx], tfidf_scores_abusive[idx]) for idx in sorted_indices_abusive[:100]]
print("Top significant 2-word collocations in abusive tweets:")
for idx, (term, score) in enumerate(top_terms_abusive, 1):
    print(f"{idx}. {term}: {score}")

#%%
def plot_word_cloud(terms_scores, title, filename):
    word_freq_dict = dict(terms_scores)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

    wordcloud.to_file(filename)

plot_word_cloud(top_terms_df, "Top significant 2-word collocations in all tweets", "new_all_tweets_wordcloud.png")

plot_word_cloud(top_terms_non_abusive, "Top significant 2-word collocations in non-abusive tweets", "new_non_abusive_tweets_wordcloud.png")

plot_word_cloud(top_terms_abusive, "Top significant 2-word collocations in abusive tweets", "new_abusive_tweets_wordcloud.png")