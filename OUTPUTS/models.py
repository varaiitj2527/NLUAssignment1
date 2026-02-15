import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix
)


def LoadAndProcessData(FilePath):
    """Loads the dataset and cleans the text data."""
    # Read the CSV file
    DataFrame = pd.read_csv(FilePath)
    
    DataFrame = DataFrame[DataFrame['Label'].isin([0,1])].copy()

    def CleanText(Text):
        """Helper function to clean individual text entries."""
        # Convert to string and lowercase
        Text = str(Text).lower()
        # Remove non-alphabetic characters using regex
        Text = re.sub(r'[^a-z\s]', '', Text)
        return Text

    # Apply the cleaning function to the 'Text' column
    DataFrame['CleanText'] = DataFrame['Text'].apply(CleanText)
    return DataFrame

def EvaluateModels(DataFrame):
    """Trains and evaluates multiple models with different feature sets."""

    InputFeatures = DataFrame['CleanText']
    TargetLabels = DataFrame['Label']

    # Split data into training and testing sets (80% train, 20% test)
    TrainFeatures, TestFeatures, TrainLabels, TestLabels = train_test_split(
        InputFeatures, TargetLabels, test_size=0.2, random_state=42
    )

    # Define different feature extraction configurations
    FeatureConfigurations = {
        'BoW (1-gram)': CountVectorizer(ngram_range=(1,1)),
        'BoW (2-gram)': CountVectorizer(ngram_range=(2,2)),
        'TF-IDF (1-gram)': TfidfVectorizer(ngram_range=(1,1)),
        'TF-IDF (2-gram)': TfidfVectorizer(ngram_range=(2,2))
    }

    # Define the machine learning models
    Models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier()
    }

    ResultsList = []

    # Iterate over each feature configuration
    for FeatureName, Vectorizer in FeatureConfigurations.items():

        print(f"\nProcessing: {FeatureName}")

        # Vectorize the text data
        TrainFeaturesVectorized = Vectorizer.fit_transform(TrainFeatures)
        TestFeaturesVectorized = Vectorizer.transform(TestFeatures)

        # Iterate over each model
        for ModelName, Model in Models.items():

            Model.fit(TrainFeaturesVectorized, TrainLabels)
            PredictedLabels = Model.predict(TestFeaturesVectorized)

            # Calculate metrics
            Accuracy = accuracy_score(TestLabels, PredictedLabels)
            F1Score = f1_score(TestLabels, PredictedLabels)
            Recall = recall_score(TestLabels, PredictedLabels)
            Precision = precision_score(TestLabels, PredictedLabels)

            print(f"\nModel: {ModelName}")
            print(classification_report(TestLabels, PredictedLabels))

            # Confusion Matrix Visualization
            ConfusionMatrix = confusion_matrix(TestLabels, PredictedLabels)
            plt.figure(figsize=(4,4))
            sns.heatmap(ConfusionMatrix, annot=True, fmt='d', cmap='Blues')
        
            TitleText = f"{ModelName} - {FeatureName}"
            plt.title(TitleText.upper())
            plt.xlabel("PREDICTED LABEL")
            plt.ylabel("ACTUAL LABEL")
            plt.show()

            # Append results to the list
            ResultsList.append({
                "Feature": FeatureName,
                "Model": ModelName,
                "Accuracy": Accuracy,
                "F1 Score": F1Score,
                "Recall": Recall,
                "Precision": Precision
            })

            # Extract and plot top words for Logistic Regression only
            if ModelName == "Logistic Regression":
                FeatureNames = Vectorizer.get_feature_names_out()
                Coefficients = Model.coef_[0]

                # Get top 10 negative words (Politics/Label 0 indicators)
                TopNegativeIndices = np.argsort(Coefficients)[:10]
                
                # Get top 10 positive words (Sports/Label 1 indicators)
                TopPositiveIndices = np.argsort(Coefficients)[-10:]
                CombinedIndices = np.hstack([TopNegativeIndices, TopPositiveIndices])

                # Extract words and weights
                TopWords = [FeatureNames[i] for i in CombinedIndices]
                TopWeights = Coefficients[CombinedIndices]

                plt.figure(figsize=(10, 8))
                
                # Create a color list: Red for Politics (Negative), Blue for Sports (Positive)
                Colors = ['red' if x < 0 else 'blue' for x in TopWeights]
                
                sns.barplot(x=TopWeights, y=TopWords, palette=Colors)
                TitleText = f"TOP WORDS (POLITICS VS SPORTS) - {FeatureName}"
                plt.title(TitleText.upper())
                plt.xlabel("COEFFICIENT WEIGHT (NEG=POLITICS, POS=SPORTS)")
                plt.ylabel("FEATURE WORD")
                plt.show()

    return pd.DataFrame(ResultsList)

def PlotAccuracy(ResultsDataFrame):
    """Plots a comparison of model accuracy across different features."""

    plt.figure(figsize=(12,6))
    sns.barplot(
        x='Feature',
        y='Accuracy',
        hue='Model',
        data=ResultsDataFrame
    )
    plt.xticks(rotation=45)
    plt.ylim(0,1.1)
    plt.title("ACCURACY COMPARISON")
    plt.xlabel("FEATURE REPRESENTATION")
    plt.ylabel("ACCURACY SCORE")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    DataFrame = LoadAndProcessData("df_file.csv")
    Results = EvaluateModels(DataFrame)

    print("\nFinal Results:")
    print(Results)
    PlotAccuracy(Results)