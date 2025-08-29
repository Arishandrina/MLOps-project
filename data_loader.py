import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Функция для загрузки и предобработки данных
def load_and_preprocess_data(file_path, output_dir="data", test_size=0.2, random_state=42):
   data = pd.read_csv(file_path)

   # Заполнение пропусков и удаление ненужных столбцов
   data['Asthma_Control_Level'] = data['Asthma_Control_Level'].fillna('No Asthma')
   data['Comorbidities'] = data['Comorbidities'].fillna('None')
   data['Allergies'] = data['Allergies'].fillna('None')
   data = data.drop('Patient_ID', axis=1)

   # Cоздание новых признаков
   data['Age_Group'] = pd.cut(data['Age'],
                              bins=[0, 12, 18, 65, 100],
                              labels=['Child', 'Teenager', 'Adult', 'Senior'],
                              right=False)
   data['BMI_Category'] = pd.cut(data['BMI'],
                                 bins=[0, 18.5, 25, 30, 100],
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                                 right=False)
  
   data = data.drop(['Age', 'BMI', 'Asthma_Control_Level'], axis=1)
   data['Environment_Risk'] = data['Occupation_Type'] + '_' + data['Air_Pollution_Level']

   # Кодирование категориальных признаков
   data_encoded = pd.get_dummies(data, drop_first=True)
   X = data_encoded.drop('Has_Asthma', axis=1)
   y = data_encoded['Has_Asthma']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

   # Масштабирование числовых признаков
   cols_to_scale = ['Medication_Adherence', 'Number_of_ER_Visits', 'Peak_Expiratory_Flow', 'FeNO_Level']
  
   scaler = StandardScaler()
   X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
   X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

   # Сохранение обработанных данных
   os.makedirs(output_dir, exist_ok=True)
   X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
   X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
   y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
   y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
  
   joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
  
   print(f"Обработанные данные сохранены в папку '{output_dir}'")

   return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
   load_and_preprocess_data("Asthma Risk & Severity Dataset.csv")