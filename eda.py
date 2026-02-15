import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def LoadAndProcessData(FilePath):
    """Reads the CSV file, filters for specific labels (0 and 1)
    and cleans the text by removing noise and stopwords
    """
    DataFrame = pd.read_csv(FilePath)
    DataFrame = DataFrame[DataFrame['Label'].isin([0, 1])].copy()

    # Create a set for faster lookup
    StopWords = set(ENGLISH_STOP_WORDS)

    def CleanText(Text):
        """Helper function to clean a single string of text"""
        # Convert to lowercase
        Text = str(Text).lower()
        # Remove non-alphabetic characters
        Text = re.sub(r'[^a-z\s]', '', Text)

        # Remove stopwords and short words (length <= 2)
        Words = [
            Word for Word in Text.split()
            if Word not in StopWords and len(Word) > 2
        ]

        return " ".join(Words)

    #Apply cleaning function to the 'Text' column
    DataFrame['CleanText'] = DataFrame['Text'].apply(CleanText)
    return DataFrame

def BasicEda(DataFrame):
    """
    Performs basic exploratory data analysis including class distribution,
    word counts, and average word lengths.
    """

    plt.figure(figsize=(6,4))
    sns.countplot(x='Label', data=DataFrame)
    
    plt.title('CLASS DISTRIBUTION: POLITICS (0) VS SPORTS (1)')
    plt.xlabel('LABEL')
    plt.ylabel('COUNT')
    plt.show()

    DataFrame['WordCount'] = DataFrame['CleanText'].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(8,5))
    sns.histplot(data=DataFrame, x='WordCount', hue='Label', kde=True)
    
    plt.title('WORD COUNT DISTRIBUTION BY CLASS')
    plt.xlabel('WORD COUNT')
    plt.ylabel('FREQUENCY')
    plt.show()

    DataFrame['AvgWordLength'] = DataFrame['CleanText'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )

    plt.figure(figsize=(6,4))
    sns.boxplot(x='Label', y='AvgWordLength', data=DataFrame)
    
    plt.title("AVERAGE WORD LENGTH PER CLASS")
    plt.xlabel('LABEL')
    plt.ylabel('AVERAGE WORD LENGTH')
    plt.show()

def PlotTopFrequentWords(DataFrame, Label, TopN=15):
    """
    Calculates and plots the most frequent words for a specific class label.
    """
    
    TextData = DataFrame[DataFrame['Label'] == Label]['CleanText']
    
    #Combine all text into one list of words
    AllWords = " ".join(TextData).split()
    WordCounter = Counter(AllWords)
    MostCommon = WordCounter.most_common(TopN)

    Words = [w[0] for w in MostCommon]
    Counts = [w[1] for w in MostCommon]

    plt.figure(figsize=(8,6))
    sns.barplot(x=Counts, y=Words)
    
    # Determine class name for title
    ClassName = 'POLITICS' if Label == 0 else 'SPORTS'
    
    # Uppercase labels and title
    plt.title(f"TOP {TopN} WORDS - {ClassName}")
    plt.xlabel("FREQUENCY")
    plt.ylabel("WORDS")
    plt.show()

def PlotWordCloud(DataFrame, Label):
    """
    Generates and displays a WordCloud for a specific class label.
    """

    # Combine all text for the label
    TextData = " ".join(DataFrame[DataFrame['Label'] == Label]['CleanText'])

    # Create WordCloud object
    WordCloudInstance = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=ENGLISH_STOP_WORDS
    )

    plt.figure(figsize=(10,5))
    plt.imshow(WordCloudInstance.generate(TextData))
    plt.axis("off")
    
    ClassName = 'POLITICS' if Label == 0 else 'SPORTS' 
    plt.title(f"WORDCLOUD - {ClassName}")
    plt.show()

if __name__ == "__main__":
    
    DataFrame = LoadAndProcessData("df_file.csv")
    BasicEda(DataFrame)

    PlotTopFrequentWords(DataFrame, Label=0)
    PlotTopFrequentWords(DataFrame, Label=1)

    PlotWordCloud(DataFrame, Label=0)
    PlotWordCloud(DataFrame, Label=1)