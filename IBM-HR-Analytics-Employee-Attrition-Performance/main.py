import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))
import pandas as pd
from config import DATA_FILE, TARGET_COLUMN, RANDOM_STATE, LASSO_FEATURES
from data_handler import DataHandler
from model_trainer import ModelTrainer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def main():

    data_handler = DataHandler(
        data_path=DATA_FILE,
        random_state=RANDOM_STATE
    )
    
    # 2. 加载和划分数据
    X, y = data_handler.load_data()
    X_train, X_test, y_train, y_test = data_handler.split_data(test_size=0.2)

    models_to_train = [
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [100, 150, 200],
                'max_depth': [4, 6],
                'min_samples_leaf': [5, 9]
            }
        },
        {
            'name': 'XGBoost',
            'model': XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [100, 150, 200],
                'max_depth': [4, 6, 8],
                'min_samples_leaf': [5, 9]
            }
        },
        {
            'name': 'LightGBM',
            'model': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            'params': {
                'learning_rate': [0.05, 0.1],
                'num_leaves': [10, 15, 31],
                'max_depth': [5, 7]
            }
        }
    ]
    
    results = {}
    
    for model_config in models_to_train:
        print(f"处理模型: {model_config['name']}")
        
        trainer = ModelTrainer(
            model_name=model_config['name'],
            base_model=model_config['model'],
            param_grid=model_config['params'],
            random_state=RANDOM_STATE
        )
        
        trainer.train(X_train, y_train)
        
        metrics = trainer.evaluate(
            X_train, y_train,
            X_test, y_test
        )
        
        trainer.plot_results(X_test, y_test)
        
        trainer.save_model(f"results/models/{model_config['name']}.pkl")
        
        results[model_config['name']] = metrics
    

    comparison = []
    for model_name, metrics in results.items():
        comparison.append({
            'Model': model_name,
            'Test_Accuracy': metrics['Test']['Accuracy'],
            'Test_Precision': metrics['Test']['Precision'],
            'Test_Recall': metrics['Test']['Recall'],
            'Test_F1': metrics['Test']['F1_Score'],
            'Test_AUC': metrics['Test']['AUC_ROC']
        })
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv("results/metrics/model_comparison.csv", index=False)

if __name__ == "__main__":
    main()