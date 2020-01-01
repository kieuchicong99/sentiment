import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False,
                  colnames=['truth', 'text']) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['truth'] = df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')
        # Optional lowercase for test data (if model was trained on lowercased text)
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['truth'], df['pred'])*100
        f1 = f1_score(df['truth'], df['pred'], average='macro')*100
        print("Accuracy: {:.3f}\nMacro F1-score: {:.3f}".format(acc, f1))

class FastTextSentiment(Base):
    """Predict sentiment scores using FastText.
    https://fasttext.cc/
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install fasttext
        import fasttext
        try:
            self.model = fasttext.load_model(model_file)
        except ValueError:
            raise Exception("Please specify a valid trained FastText model file (.bin or .ftz extension)'{}'."
                            .format(model_file))

    def score(self, text: str) -> int:
        # Predict just the top label (hence 1 index below)
        labels, probabilities = self.model.predict(text, 1)
        pred = int(labels[0][-1])
        return pred

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].apply(self.score)
        return df


class FlairSentiment(Base):
    """Predict sentiment scores using Flair.
    https://github.com/zalandoresearch/flair
    Tested on Flair version 0.4.2+ and Python 3.6+
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        "Use the latest version of Flair NLP from their GitHub repo!"
        # pip install flair
        from flair.models import TextClassifier
        try:
            self.model = TextClassifier.load(model_file)
        except ValueError:
            raise Exception("Please specify a valid trained Flair PyTorch model file (.pt extension)'{}'."
                            .format(model_file))

    def score(self, text: str) -> int:
        from flair.data import Sentence
        doc = Sentence(text)
        self.model.predict(doc)
        pred = int(doc.labels[0].value)
        return pred

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Use tqdm to display model prediction status bar"
        # pip install tqdm
        from tqdm import tqdm
        tqdm.pandas()
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].progress_apply(self.score)
        return df

