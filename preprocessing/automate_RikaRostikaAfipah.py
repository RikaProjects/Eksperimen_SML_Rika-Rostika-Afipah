"""
Automated Data Preprocessing for Heart Disease Dataset
Nama: Rika Rostika Afipah
Date: 02 November 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os
import urllib.request
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeartDiseasePreprocessor:
    def __init__(self, data_path=None):
        """
        Initialize preprocessor with data path
        
        Args:
            data_path (str): Path to raw data file. If None, will download from UCI
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        
        # Column names for heart disease dataset
        self.column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        # UCI Dataset URL
        self.uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    def download_data(self, download_path='../heart_disease_raw/'):
        """Download data from UCI repository if not available locally"""
        logger.info("🌐 Downloading data from UCI repository...")
        
        # Create directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        download_file_path = os.path.join(download_path, 'processed.cleveland.data')
        
        try:
            urllib.request.urlretrieve(self.uci_url, download_file_path)
            logger.info("✅ Data downloaded successfully to: %s", download_file_path)
            self.data_path = download_file_path
            return download_file_path
        except Exception as e:
            logger.error("❌ Failed to download data: %s", e)
            raise
    
    def load_data(self):
        """Load and initial data inspection"""
        logger.info("📥 Loading data from %s", self.data_path)
        
        try:
            # If data_path is not provided or file doesn't exist, download it
            if self.data_path is None or not os.path.exists(self.data_path):
                logger.warning("📥 Local data file not found, downloading from UCI...")
                self.data_path = self.download_data()
            
            self.data = pd.read_csv(self.data_path, names=self.column_names, na_values='?')
            logger.info("✅ Data loaded successfully. Shape: %s", self.data.shape)
            
            # Basic info
            logger.info("📊 Data columns: %s", list(self.data.columns))
            logger.info("🎯 Target distribution: \n%s", self.data['target'].value_counts().sort_index())
            
        except Exception as e:
            logger.error("❌ Failed to load data: %s", e)
            raise
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        logger.info("🔍 Handling missing values...")
        
        missing_before = self.data.isnull().sum()
        if missing_before.sum() > 0:
            logger.info("Missing values found: \n%s", missing_before[missing_before > 0])
            
            # Fill numerical missing values with median
            numerical_cols = ['ca', 'thal']
            for col in numerical_cols:
                if col in self.data.columns and self.data[col].isnull().sum() > 0:
                    median_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_val)
                    logger.info("✅ Filled missing values in %s with median: %s", col, median_val)
        else:
            logger.info("✅ No missing values found")
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        duplicates = self.data.duplicated().sum()
        logger.info("🔍 Found %s duplicate rows", duplicates)
        
        if duplicates > 0:
            self.data = self.data.drop_duplicates()
            logger.info("✅ Removed duplicates. New shape: %s", self.data.shape)
    
    def create_heart_disease_target(self):
        """Convert multi-class target to binary classification"""
        logger.info("🎯 Creating binary target variable...")
        
        # Convert to binary: 0 = no disease, >0 = disease
        self.data['heart_disease'] = (self.data['target'] > 0).astype(int)
        logger.info("Binary target distribution: \n%s", self.data['heart_disease'].value_counts())
    
    def feature_engineering(self):
        """Create new features from existing ones"""
        logger.info("🔧 Performing feature engineering...")
        
        # Categorize age
        def categorize_age(age):
            if age < 45:
                return 'young'
            elif age < 60:
                return 'middle'
            else:
                return 'senior'
        
        self.data['age_category'] = self.data['age'].apply(categorize_age)
        logger.info("Age categories: \n%s", self.data['age_category'].value_counts())
        
        # Categorize cholesterol
        def categorize_chol(chol):
            if chol < 200:
                return 'normal'
            elif chol < 240:
                return 'borderline'
            else:
                return 'high'
        
        self.data['chol_category'] = self.data['chol'].apply(categorize_chol)
        logger.info("Cholesterol categories: \n%s", self.data['chol_category'].value_counts())
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        logger.info("🔤 Encoding categorical features...")
        
        categorical_cols = ['age_category', 'chol_category']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
            label_encoders[col] = le
            logger.info("%s encoding: %s", col, list(zip(le.classes_, le.transform(le.classes_))))
        
        return label_encoders
    
    def handle_outliers(self):
        """Handle outliers using IQR method"""
        logger.info("📊 Handling outliers...")
        
        # Select numerical columns (excluding target)
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['target', 'heart_disease']]
        
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
            if outliers > 0:
                logger.info("Outliers in %s: %s (%.2f%%)", col, outliers, outliers/len(self.data)*100)
                
                # Cap outliers
                self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
    
    def prepare_final_data(self):
        """Prepare final dataset for modelling"""
        logger.info("🔧 Preparing final dataset...")
        
        # Drop columns not needed for modelling
        columns_to_drop = ['target', 'age_category', 'chol_category']
        self.processed_data = self.data.drop(columns_to_drop, axis=1)
        
        logger.info("✅ Final dataset shape: %s", self.processed_data.shape)
        logger.info("🎯 Target variable: heart_disease")
        logger.info("📈 Features: %s", list(self.processed_data.columns))
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        logger.info("📊 Splitting data into training and testing sets...")
        
        X = self.processed_data.drop('heart_disease', axis=1)
        y = self.processed_data['heart_disease']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info("Training set: %s", X_train.shape)
        logger.info("Testing set: %s", X_test.shape)
        logger.info("Training target distribution: %s", y_train.value_counts().to_dict())
        logger.info("Testing target distribution: %s", y_test.value_counts().to_dict())
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, output_path):
        """Save processed data to file"""
        logger.info("💾 Saving processed data to %s", output_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False)
        logger.info("✅ Processed data saved successfully")
    
    def run_pipeline(self, output_path):
        """Run complete preprocessing pipeline"""
        logger.info("🚀 Starting automated preprocessing pipeline...")
        
        try:
            # Step 1: Load data (will download if not available)
            self.load_data()
            
            # Step 2: Handle missing values
            self.handle_missing_values()
            
            # Step 3: Remove duplicates
            self.remove_duplicates()
            
            # Step 4: Create target variable
            self.create_heart_disease_target()
            
            # Step 5: Feature engineering
            self.feature_engineering()
            
            # Step 6: Encode categorical features
            self.encode_categorical_features()
            
            # Step 7: Handle outliers
            self.handle_outliers()
            
            # Step 8: Prepare final data
            self.prepare_final_data()
            
            # Step 9: Save processed data
            self.save_processed_data(output_path)
            
            # Step 10: Split data
            X_train, X_test, y_train, y_test = self.split_data()
            
            logger.info("🎉 Preprocessing pipeline completed successfully!")
            
            return {
                'processed_data': self.processed_data,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'preprocessor': self
            }
            
        except Exception as e:
            logger.error("❌ Preprocessing pipeline failed: %s", e)
            raise

def main():
    """Main function to run the preprocessing pipeline"""
    
    # Configuration - menggunakan path yang lebih reliable
    OUTPUT_PATH = './heart_disease_processed/heart_disease_processed.csv'
    
    # Initialize preprocessor - tanpa path, akan download otomatis
    preprocessor = HeartDiseasePreprocessor()
    
    # Run pipeline
    results = preprocessor.run_pipeline(OUTPUT_PATH)
    
    print("\n" + "="*50)
    print("📋 PREPROCESSING SUMMARY")
    print("="*50)
    print(f"✅ Raw data shape: {preprocessor.data.shape}")
    print(f"✅ Processed data shape: {preprocessor.processed_data.shape}")
    print(f"✅ Training set: {results['X_train'].shape}")
    print(f"✅ Testing set: {results['X_test'].shape}")
    print(f"✅ Output saved to: {OUTPUT_PATH}")
    print("="*50)
    
    return results

if __name__ == "__main__":
    results = main()