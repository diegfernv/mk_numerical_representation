import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from joblib import dump

class Models():
    algorithms = [
        ("knn", KNeighborsClassifier()),
        ("dt", DecisionTreeClassifier()),
        ("svm", SVC()),
        ("rf", RandomForestClassifier()),
        ("ada", AdaBoostClassifier()) 
    ]
    
    def train_all(self, data, response):
        for name, algorithm in self.algorithms:
            print("Training with", name)
            algorithm.fit(X=data, y=response)

    def save(self, sufix):
        for name, algorithm in self.algorithms:
            dump(algorithm, f"models/{name}_model_{sufix}.joblib")

    def predict(self, X_val):
        predictions = []

        for name, model in self.algorithms:
            predictions.append((name, model.predict(X_val)))
        
        return predictions

    def predict_proba(self, X_val):
        predictions = []

        for name, model in self.algorithms:
            predictions.append((name, model.predict_proba(X_val)[::,1]))
        
        return predictions
    
    def cross_validate_model(self, name, model, X, y, cv, scoring):
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        return [name, res["test_accuracy"].mean(), res["test_precision"].mean(), res["test_recall"].mean(), res["test_f1"].mean(), res["test_mcc"].mean()]
    
    def cross_validate(self, X, y, cv, scoring):
        score_list = []
        for name, model in self.algorithms:
            score_list.append(self.cross_validate_model(name, model, X, y, cv, scoring))
        
        return pd.DataFrame(score_list, columns=["algorithm", "accuracy", "precision", "recall", "f1", "mcc"])

    def cv_predict(self, X, y, cv):
        self.predictions = cross_val_predict(self.model_instance, X, y, cv=cv)
        return self.get_metrics(y)

    def get_metrics(self, X_val,y_true):
        export_list = []
        predictions = self.predict(X_val)
        for name, y_pred in predictions:
            acc_value = accuracy_score(y_pred=y_pred, y_true=y_true) 
            recall_value = recall_score(y_pred=y_pred, y_true=y_true)
            precision_value = precision_score(y_pred=y_pred, y_true=y_true) 
            f1_value = f1_score(y_pred=y_pred, y_true=y_true)
            mcc_value = matthews_corrcoef(y_pred=y_pred, y_true=y_true)
            cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
            export_list.append((name, acc_value, recall_value, precision_value, f1_value, mcc_value, cm))
    
        return pd.DataFrame(export_list, columns=["algorithm", "acc", "recall", "precision", "f1", "mcc", "cm"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, help="Input file")
    parser.add_argument("-o","--output", type=str, help="Output path")
    parser.add_argument("-n","--name", type=str, help="File name")
    args = parser.parse_args()

    print(f"Loading data {args.input}")
    df = pd.read_csv(args.input)
    X = df.drop('response', axis=1)
    y = df['response']

    print("Splitting data")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    models_instance = Models()
    models_instance.train_all(X_train, y_train)

    print("Getting metrics")
    metrics_val = models_instance.get_metrics(X_val, y_val)
    metrics_test = models_instance.get_metrics(X_test, y_test)
    metrics = pd.merge(metrics_val, metrics_test, on="algorithm", suffixes=("_val", "_test"))
    metrics["encoding"] = args.input.split(".")[0].split("/")[-1]

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if os.path.exists(f"{args.output}/{args.name}_val_test.csv"):
        output_df = pd.read_csv(f"{args.output}/{args.name}_val_test.csv")
        output_df = pd.concat([output_df, metrics], axis=0)
    else:
        output_df = metrics

    output_df.to_csv(f"{args.output}/{args.name}_val_test.csv", index=False)

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "mcc": make_scorer(matthews_corrcoef)
    }

    print("Cross validating")
    cv_metrics = models_instance.cross_validate(X_train, y_train, cv=5, scoring=scoring)
    cv_metrics["encoding"] = args.input.split(".")[0].split("/")[-1]

    if os.path.exists(f"{args.output}/{args.name}_cv.csv"):
        output_df = pd.read_csv(f"{args.output}/{args.name}_cv.csv")
        output_df = pd.concat([output_df, cv_metrics], axis=0)
    else:
        output_df = cv_metrics

    output_df.to_csv(f"{args.output}/{args.name}_cv.csv", index=False)
