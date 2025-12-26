import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DataHandler:
    '''数据处理类'''
    def __init__(self, data_path=None, target_col='Attrition_yes', random_state=42, verbose=True):
        self.data_path = data_path
        self.target_col = target_col
        self.random_state = random_state
        self.verbose = verbose
        
        # 初始化数据容器
        self.data = None
        self.X = None
        self.y = None     
        self.X_train = None  
        self.X_test = None
        self.y_train = None 
        self.y_test = None

        if verbose and data_path:
            print(f"数据处理器初始化: 目标变量={target_col}")
    
    def load_data(self):
        if not self.data_path:
            print("错误: 未提供数据路径")
            return None, None
        
        # 读取数据
        self.data = pd.read_csv(self.data_path)
        # 特征与目标变量
        self.X = self.data.drop(self.target_col, axis=1)
        self.y = self.data[self.target_col]
        
        if self.verbose:
            print(f"数据加载成功")
            print(f"数据形状: {self.data.shape}")
            print(f"目标变量分布:")
            for value, count in Counter(self.y).items():
                percent = count / len(self.y) * 100
                print(f" - {value}: {count} ({percent:.1f}%)")
        return self.X, self.y
    
    def split_data(self, test_size=0.2):
        if self.X is None or self.y is None:
            print("错误: 请先加载数据")
            return None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=self.y
        )
        
        if self.verbose:
            print(f"数据划分完成")
            print(f"训练集: {self.X_train.shape}")
            print(f"测试集: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_smote(self, sampling_strategy=0.85):
        if self.X_train is None or self.y_train is None:
            print("错误: 请先划分数据")
            return None
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        
        if self.verbose:
            print(f"SMOTE重采样完成")
            print(f"原始: {Counter(self.y_train)}")
            print(f"重采样后: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def select_features(self, feature_list):
        if self.X_train is None:
            print("错误: 请先划分数据")
            return None
        
        # 检查哪些特征存在
        available_features = []
        for feature in feature_list:
            if feature in self.X_train.columns:
                available_features.append(feature)
            elif self.verbose:
                print(f"警告: 特征 '{feature}' 不存在")
        
        # 选择特征
        X_train_selected = self.X_train[available_features]
        X_test_selected = self.X_test[available_features]
        
        if self.verbose:
            print(f"特征选择完成")
            print(f"选择了 {len(available_features)} 个特征")
        
        return X_train_selected, X_test_selected
    
    def get_info(self):
        info = {
            '数据已加载': self.data is not None,
            '数据已划分': self.X_train is not None,
            '目标变量': self.target_col
        }
        
        if self.data is not None:
            info['数据形状'] = self.data.shape
            info['特征数量'] = self.X.shape[1]
        
        if self.X_train is not None:
            info['训练集大小'] = self.X_train.shape
            info['测试集大小'] = self.X_test.shape
        
        return info