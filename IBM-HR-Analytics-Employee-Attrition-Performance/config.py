import os
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'


DATA_FILE = DATA_DIR / 'HR_cleaned_scaled.csv'
TARGET_COLUMN = 'Attrition_yes'

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.125  # 验证集占训练集的比例

# LASSO特征筛选结果
LASSO_FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'EmployeeNumber', 
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 
    'BusinessTravel_travel_frequently', 'BusinessTravel_travel_rarely', 
    'Department_research & development', 'Gender_male', 
    'JobRole_laboratory technician', 'JobRole_research director', 
    'JobRole_sales representative', 'MaritalStatus_married', 
    'MaritalStatus_single', 'OverTime_yes', 'EducationField_life sciences', 
    'EducationField_marketing', 'EducationField_medical', 'EducationField_other', 
    'EducationField_technical degree'
]

# 确保目录存在
for directory in [DATA_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)