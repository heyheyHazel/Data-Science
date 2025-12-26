import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)

class ModelTrainer:    
    def __init__(self, model_name, base_model, param_grid, random_state = 42):
        self.model_name = model_name
        self.base_model = base_model
        self.param_grid = param_grid
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.train_metrics = None
        self.test_metrics = None
        
    def train(self, X_train, y_train, cv = 5, scoring = 'roc_auc'):
        print(f"开始训练: {self.model_name}")

        # 交叉验证
        cv_strategy = StratifiedKFold(
            n_splits=cv, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=self.base_model,
            param_grid=self.param_grid,
            scoring=scoring,
            cv=cv_strategy,
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"最佳参数: {self.best_params}")
        print(f"最佳{scoring}: {self.best_score:.4f}")
        
        return self.best_model
    
    def evaluate(self, X_train, y_train, X_test, y_test, plot=True):

        if self.best_model is None:
            print("错误: 请先训练模型")
            return None
        
        results = {'Train': {}, 'Test': {}}
        
        # 对训练集和测试集分别评估
        for name, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
            y_pred = self.best_model.predict(X)
            
            results[name] = {
                'Accuracy': accuracy_score(y, y_pred),
                'Precision': precision_score(y, y_pred, zero_division=0),
                'Recall': recall_score(y, y_pred, zero_division=0),
                'F1_Score': f1_score(y, y_pred, zero_division=0)
            }
            
            # 尝试计算AUC-ROC
            try:
                y_prob = self.best_model.predict_proba(X)[:, 1]
                results[name]['AUC_ROC'] = roc_auc_score(y, y_prob)
                results[name]['y_prob'] = y_prob
            except:
                results[name]['AUC_ROC'] = 0
                results[name]['y_prob'] = None
            
            results[name]['y_pred'] = y_pred
            
            # 打印结果
            print(f"\n{name}集:")
            for metric, value in results[name].items():
                if metric not in ['y_pred', 'y_prob']:
                    print(f"  {metric}: {value:.4f}")
        
        self.metrics = results
        return results
    
    def plot_results(self, X_test, y_test, save_dir = "results/plots"):
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5), dpi = 300)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.xlabel('Predict')
        plt.ylabel('True')
        plt.savefig(f"{save_dir}/{self.model_name}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. ROC曲线
        if hasattr(self.best_model, 'predict_proba'):
            y_prob = self.best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'{self.model_name} - ROC')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
            plt.savefig(f"{save_dir}/{self.model_name}_roc_curve.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_model(self, filepath: str):
        joblib.dump(self.best_model, filepath)
        print(f"模型已保存到: {filepath}")