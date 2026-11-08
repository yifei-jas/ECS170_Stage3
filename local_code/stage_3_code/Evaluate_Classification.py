"""Multiclass evaluation with the dataset's actual class count."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from local_code.base_class.evaluate import evaluate


class Evaluate_Classification(evaluate):
    def __init__(self, classes):
        super().__init__('Classification metrics','Macro and weighted metrics')
        self.classes=classes

    def evaluate(self):
        truth,pred=self.data['true_y'],self.data['pred_y']
        scores={'accuracy':float(accuracy_score(truth,pred))}
        for average in ('macro','weighted'):
            precision,recall,f1,_=precision_recall_fscore_support(
                truth,pred,labels=list(range(self.classes)),average=average,zero_division=0)
            scores.update({f'precision_{average}':float(precision),
                           f'recall_{average}':float(recall),f'f1_{average}':float(f1)})
        scores['confusion_matrix']=confusion_matrix(truth,pred,labels=list(range(self.classes))).tolist()
        return scores
