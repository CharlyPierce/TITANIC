
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,  MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from google.cloud import storage
from joblib import dump, load
from datetime import datetime
import os


class DataSetPreparation(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Definimos las categorías para cada columna
        self.name_categories = ['BUCKET_MR', 'BUCKET_MISS', 'BUCKET_MRS', 'BUCKET_NONE']
        self.sex_categories = ['f', 'M', 'O']
        self.embarked_categories = ['s', 'c', 'q', 'o']
        self.cabin_level_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'U']

        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.imputer_2 = SimpleImputer(strategy='most_frequent')

        # Columns
        self.BINARY_FEATURES = ['Sex']
        self.NUMERIC_FEATURES = ['Age', 'Fare']
        self.CATEGORICAL_FEATURES = ['Pclass', 'Embarked','Parch', 'SibSp']

    def fit(self, X, y=None):
        X = X.copy()

        # Imputar valores faltantes para columnas excepto 'Age'
        for col in ['Pclass', 'SibSp', 'Parch', 'Fare']:
            X[col] = self.imputer.fit_transform(X[[col]])

        # Calcular media y desviación estándar (no sesgada) de la edad
        self.age_mean = X['Age'].mean()
        self.age_std = X['Age'].std(ddof=1)  # ddof=1 para desviación estándar no sesgada

        # Contar cuántos valores faltantes de edad hay
        missing_age_count = X['Age'].isnull().sum()

        # Generar una muestra aleatoria de edades basada en la media y desviación estándar
        age_sample = np.random.normal(self.age_mean, self.age_std, missing_age_count)

        # Rellenar valores faltantes de edad con la muestra generada
        age_series = X['Age'].copy()
        age_series[np.isnan(age_series)] = age_sample
        X['Age'] = age_series

        X["Embarked"] = self.imputer_2.fit_transform(X[["Embarked"]]).ravel()


        # Ajustar el escalador de estandarización
        self.std_scaler.fit(X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
        # Transformar los datos con el escalador de estandarización y ajustar el escalador MinMax
        self.minmax_scaler.fit(X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

        self.ticket_frequencies = X['Ticket'].fillna("Unknown").value_counts()
        self.cabin_frequencies = X['Cabin'].value_counts()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('bin', OrdinalEncoder(), self.BINARY_FEATURES),
                ('num', StandardScaler(), self.NUMERIC_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.CATEGORICAL_FEATURES)
            ]
        )

        self.preprocessor.fit(X)

        return self

    def transform(self, X, y=None):
        # Borrar duplicados
        X = X.drop_duplicates().copy()
        missing_age_count = X['Age'].isnull().sum()
        age_sample = np.random.normal(self.age_mean, self.age_std, missing_age_count)
        age_series = X['Age'].copy()
        age_series[np.isnan(age_series)] = age_sample
        X['Age'] = age_series
        # Rellenar valores faltantes para las columnas numéricas con las medianas "aprendidas" durante el fit
        for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
            X[col].fillna(-1, inplace=True)
        X["Embarked"] = self.imputer_2.transform(X[["Embarked"]]).ravel() #ravel convierte 2d en 1d

        for col in ["Name", "Sex", "Ticket", "Cabin"]:
            X[col].fillna("Unknown", inplace=True)

##-----------------------VARIABLE CATEGORICAS AUN -------------------------------------------##
        X['NAME_BUCKET'] = X['Name'].apply(self.categorize_name)       #4 VALORES POSIBLES
        X['SEX_BUCKET']  = X['Sex'].apply(self.categorize_sex)         #3 VALORES POSIBLES
        X['EMBARKED']  = X['Embarked'].apply(self.categorize_embarked) #4 VALORES POSIBLES

        # FEATURE SINTETIC
        X['CABINLEVEL'] = X['Cabin'].str[0].fillna("Unknown")          #8 VALORES POSIBLES

###---------------------------------------------VARIABLES FLOAT -------------------------------------------------------------------##
        # Transformar usando estandarización y guardar en nuevas columnas
        X["AGE"] = X["Age"]
###----------------------------towards------------------
        data = [X]
        for dataset in data:
            dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
            dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
            dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
            dataset['not_alone'] = dataset['not_alone'].astype(int)
        deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
        for dataset in data:
            dataset['CABIN'] = dataset['Cabin'].fillna("U0")
            dataset['Deck'] = dataset['CABIN'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
            dataset['Deck'] = dataset['Deck'].map(deck)
            dataset['Deck'] = dataset['Deck'].fillna(0)
            dataset['Deck'] = dataset['Deck'].astype(int)
        X = X.drop(['CABIN'], axis=1)
        for dataset in data:
            mean = self.age_mean
            std = self.age_std
            is_null = dataset["Age"].isnull().sum()
            # compute random numbers between the mean, std and is_null
            rand_age = np.random.randint(mean - std, mean + std, size=is_null)
            # fill NaN values in Age column with random values generated
            age_slice = dataset["Age"].copy()
            age_slice[np.isnan(age_slice)] = rand_age
            dataset["Age"] = age_slice
            dataset["AGE2"] = dataset["Age"].astype(int)
        common_value = 'S'
        for dataset in data:
            dataset['EMBARKED'] = dataset['Embarked'].fillna(common_value)
        for dataset in data:
            dataset['FARE'] = dataset['Fare'].fillna(0)
            dataset['FARE'] = dataset['FARE'].astype(int)
        titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        for dataset in data:
            # extract titles
            dataset['TITLE'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
            # replace titles with a more common title or as Rare
            dataset['TITLE'] = dataset['TITLE'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                                    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            dataset['TITLE'] = dataset['TITLE'].replace('Mlle', 'Miss')
            dataset['TITLE'] = dataset['TITLE'].replace('Ms', 'Miss')
            dataset['TITLE'] = dataset['TITLE'].replace('Mme', 'Mrs')
            # convert titles into numbers
            dataset['TITLE'] = dataset['TITLE'].map(titles)
            # filling NaN with 0, to get safe
            dataset['TITLE'] = dataset['TITLE'].fillna(0)
        genders = {"male": 0, "female": 1}
        for dataset in data:
            dataset['SEX'] = dataset['Sex'].map(genders)
        ports = {"S": 0, "C": 1, "Q": 2}
        for dataset in data:
            dataset['EMBARKED'] = dataset['EMBARKED'].map(ports)
        for dataset in data:
            dataset['AGE2'] = dataset['AGE2'].astype(int)
            dataset.loc[ dataset['AGE2'] <= 11, 'AGE2'] = 0
            dataset.loc[(dataset['AGE2'] > 11) & (dataset['AGE2'] <= 18), 'AGE2'] = 1
            dataset.loc[(dataset['AGE2'] > 18) & (dataset['AGE2'] <= 22), 'AGE2'] = 2
            dataset.loc[(dataset['AGE2'] > 22) & (dataset['AGE2'] <= 27), 'AGE2'] = 3
            dataset.loc[(dataset['AGE2'] > 27) & (dataset['AGE2'] <= 33), 'AGE2'] = 4
            dataset.loc[(dataset['AGE2'] > 33) & (dataset['AGE2'] <= 40), 'AGE2'] = 5
            dataset.loc[(dataset['AGE2'] > 40) & (dataset['AGE2'] <= 66), 'AGE2'] = 6
            dataset.loc[ dataset['AGE2'] > 66, 'AGE2'] = 6
        for dataset in data:
            dataset.loc[ dataset['FARE'] <= 7.91, 'FARE'] = 0
            dataset.loc[(dataset['FARE'] > 7.91) & (dataset['FARE'] <= 14.454), 'FARE'] = 1
            dataset.loc[(dataset['FARE'] > 14.454) & (dataset['FARE'] <= 31), 'FARE']   = 2
            dataset.loc[(dataset['FARE'] > 31) & (dataset['FARE'] <= 99), 'FARE']   = 3
            dataset.loc[(dataset['FARE'] > 99) & (dataset['FARE'] <= 250), 'FARE']   = 4
            dataset.loc[ dataset['FARE'] > 250, 'FARE'] = 5
            dataset['FARE'] = dataset['FARE'].astype(int)
        for dataset in data:
            dataset['Age_Class']= dataset['AGE2']* dataset['Pclass']
        for dataset in data:
            dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
            dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
        # Crear columnas basadas en diferentes cuantiles 
        data_transformed = self.preprocessor.transform(X)
        df_transformed = pd.DataFrame(data_transformed)
        X = pd.concat([X.reset_index(drop=True), df_transformed.reset_index(drop=True)], axis=1)
        X.loc[:, ['Pclass_std', 'Age_std', 'SibSp_std', 'Parch_std', 'Fare_std']] = self.std_scaler.transform(X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
        # Transformar usando MinMaxScaler y guardar en otras nuevas columnas
        X.loc[:, ['Pclass_mm', 'Age_mm', 'SibSp_mm', 'Parch_mm', 'Fare_mm']] = self.minmax_scaler.transform(X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
    ##-------------------FEATURE ENGINEER A VARIABLES CATEGORICAS BRUTAS  -> FLOAT/INT---------------------------------------------------------#
        X['TICKET_BUCKET']  = X['Ticket'].apply(self.categorize_ticket)
        X['GROUPSIZE'] = X['Ticket'].map(self.ticket_frequencies).fillna(1)
        # Paso 2: Ajustar la columna "GROUPSIZE" con SibSp y Parch
        X['GROUPSIZE'] = X.apply(lambda row: max(row['GROUPSIZE'], row['SibSp'] + row['Parch'] + 1), axis=1)
        # Paso 3: Crear una columna "IsAlone"
        X['ISALONE'] = (X['GROUPSIZE'] == 1).astype(int)
        # FEATURE SINTETIC
        X['CABINGROUPSIZE'] = X['Cabin'].map(self.cabin_frequencies).fillna(-1)#4 VALORES POSIBLES
##-----------------------FEATURE ENGINEER A VARIABLES CATEGORICAS LIMPIADAS -> FLOAT-------------------------------------------##
        # Aplica la función custom_ohe a cada columna
        name_ohe = self.custom_ohe(X['NAME_BUCKET'], self.name_categories)
        sex_ohe = self.custom_ohe(X['SEX_BUCKET'], self.sex_categories)
        embarked_ohe = self.custom_ohe(X['EMBARKED'], self.embarked_categories)
        cabin_ohe = self.custom_ohe(X['CABINLEVEL'], self.cabin_level_categories)

        # Combina los DataFrames resultantes
        X.columns = X.columns.astype(str)
        ohe_result = pd.concat([name_ohe, sex_ohe, embarked_ohe, cabin_ohe], axis=1)
        X = pd.concat([X, ohe_result], axis=1)
        return X

    @staticmethod
    def categorize_name(name):
        name_lower = name.lower()
        if 'mr' in name_lower:
            return 'BUCKET_MR'
        elif 'miss' in name_lower or 'mlle' in name_lower or 'ms' in name_lower:
            return 'BUCKET_MISS'
        elif 'mrs' in name_lower or 'Mme' in name_lower:
            return 'BUCKET_MRS'
        else:
            return 'BUCKET_NONE'
    @staticmethod
    def categorize_sex(sex):
        sex_lower = sex.lower()
        if 'female' in sex_lower:
            return 'f'
        elif 'male' in sex_lower:
            return 'M'
        else:
            return 'O'
    def categorize_ticket(self, ticket):
        freq = self.ticket_frequencies.get(ticket, 0)
        return freq if 1 <= freq <= 7 else 0
    # Función para asignar cuantiles
    @staticmethod
    def assign_quantile(df, column, n_quantiles):
        labels = list(range(1, n_quantiles + 1))
        return pd.qcut(df[column], q=n_quantiles, labels=labels, retbins=True, duplicates='drop') 
    @staticmethod
    def categorize_embarked(em):
        em_lower = em.lower()
        if 's' in em_lower:
            return 's'
        elif 'c' in em_lower:
            return 'c'
        elif 'q' in em_lower:
            return 'q'
        else:
            return 'o'
    def custom_ohe(self, column, categories):
        """Aplica One-Hot Encoding a la columna dadas las categorías especificadas."""
        # Crea un DataFrame con OHE
        ohe_df = pd.get_dummies(column, prefix_sep='_')

        # Crea un DataFrame vacío con las categorías deseadas
        custom_ohe_df = pd.DataFrame(0, index=ohe_df.index, columns=categories)

        # Llena el DataFrame personalizado con las columnas del OHE original
        for col in categories:
            if col in ohe_df.columns:
                custom_ohe_df[col] = ohe_df[col].values

        return custom_ohe_df
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Clean missing numerics
        for n in self.NUMERIC_FEATURES:
            df[n] = pd.to_numeric(df[n], errors='coerce')
        df = df.fillna(df.mean())

        # 2. Transformation Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('bin', OrdinalEncoder(), self.BINARY_FEATURES),
                ('num', StandardScaler(), self.NUMERIC_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.CATEGORICAL_FEATURES)
            ]
        )

        data_transformed = preprocessor.fit_transform(df)

        # Convert transformed data back to DataFrame for readability (optional)
        columns_transformed = (self.BINARY_FEATURES +
                              self.NUMERIC_FEATURES +
                              list(preprocessor.named_transformers_['cat'].get_feature_names_out(self.CATEGORICAL_FEATURES)))

        df_transformed = pd.DataFrame(data_transformed, columns=columns_transformed)
        return df_transformed



class feature_engineer(BaseEstimator, TransformerMixin):
    def fit(self, data_, y=None):
        # Obtener el índice de la columna "AGE"
        idx_age = data_.columns.get_loc("AGE")
        # Seleccionar todas las columnas desde "AGE" en adelante
        data = data_.iloc[:, idx_age:]
        data_without_missing_age = data[data['AGE'] != -1]
        data_for_clustering = data_without_missing_age.drop(columns=['Age_std', 'Age_mm'])

        data_for_clustering_all = data.drop(columns=['AGE','Age_std', 'Age_mm']) #Agregue esta linea con todos los datos para calcular al final
        data_for_clustering_all_2 = data.drop(columns=['Age_std', 'Age_mm']) #Agregue esta linea con todos los datos para calcular al final

        self.c1 = data_for_clustering_all.columns
        self.c2 = data_for_clustering_all_2.columns

        intervals = [(i, i+5) for i in range(0, 85, 5)]
        selected_centroids = []
        # 2. Seleccionar un dato aleatorio dentro de cada intervalo
        np.random.seed(42)
        for start, end in intervals:
            subset = data_without_missing_age[(data_without_missing_age['AGE'] >= start) & (data_without_missing_age['AGE'] < end)]
            if not subset.empty:
                sample = subset.sample(1).iloc[:,1:]
                selected_centroids.append(sample)
        # # 3. Concatenar todos los datos seleccionados
        initial_centroids_df = pd.concat(selected_centroids)
        # # Definiendo las características para clustering
        # Obtener todas las columnas de 'data'
        features_for_clustering = data.columns.tolist()
        # Excluir las columnas 'AGE', 'Age_std' y 'Age_mm'
        features_for_clustering = [column for column in features_for_clustering if column not in ['AGE', 'Age_std', 'Age_mm']]
        initial_centroids_values = initial_centroids_df[features_for_clustering].values
        # # Aplicar KMeans
        self.kmeans  = KMeans(n_clusters=len(initial_centroids_values), init=initial_centroids_values, n_init=1, random_state=42).fit(data_for_clustering.iloc[:,1:])
        self.kmeans_2 = KMeans(n_clusters=20, n_init=10, random_state=42).fit(data_for_clustering)
        self.kmeans_models = [KMeans(n_clusters=int(i+2), n_init=10, random_state=i).fit(data) for i in range(15)]

        return self

    def transform(self, data_):
        # Aquí pondrás todas las transformaciones adicionales que necesitas.
        idx_age = data_.columns.get_loc("AGE")
        # Seleccionar todas las columnas desde "AGE" en adelante
        data_a = data_.iloc[:, idx_age:]

        # Predecir los clusters
        cluster_assignments = self.kmeans.predict(data_[self.c1])
        cluster_assignments_2 = self.kmeans_2.predict(data_[self.c2])
        # Agregar las asignaciones de clusters como nuevas columnas
        data_['Cluster_1'] = cluster_assignments
        data_['Cluster_2'] = cluster_assignments_2

        for i, model in enumerate(self.kmeans_models):
          column_name = f'Cluster_{i+3}'
          data_[column_name] = model.predict(data_a)

        return data_

class PredictData(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.best_model = None
        self.columns_ = None
        if os.path.exists("../key.json"): os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"

    def fit(self, data_, y=None):
        self.load_model()
        idx_age = data_.columns.get_loc("AGE")
        self.columns_ = data_.columns[idx_age:]
        # Seleccionar todas las columnas desde "AGE" en adelante
        x_train_, y_train_ = data_.iloc[:, idx_age:], data_.iloc[:, 1]
        if self.best_model is None:
            param_grid = {
                'n_estimators': [100,200,300],
                'max_depth': [20,40,80],
                'min_samples_split': [2,15,30],
                'min_samples_leaf': [1,5,50,100]
            }
            grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
            grid_search.fit(x_train_, y_train_)
            self.best_model = grid_search.best_estimator_
        else:
            # Si ya hay un modelo cargado, ajusta el modelo al nuevo conjunto de datos
            self.best_model.fit(x_train_, y_train_)
        # Guarda el modelo entrenado localmente
        dump(self.best_model, 'model.joblib')
        # Sube el modelo entrenado a GCS
        try:
            self.save_model()
        except Exception as e:
            print(f"Error al subir el modelo entrenado a GCS: {str(e)}")
        return self

    def transform(self, data_):
      idx_age = data_.columns.get_loc("AGE")
      self.columns_ = data_.columns[idx_age:]
      # Seleccionar todas las columnas desde "AGE" en adelante
      x_train_ = data_.iloc[:,idx_age:]

      y_pred = self.best_model.predict(x_train_)
      return y_pred
    def save_model(self):
      current_date = datetime.now().strftime('%Y-%m-%d')
      current_hour = datetime.utcnow().strftime('%H:%M:%S')
      # Initialize the GCS client
      gcs_client = storage.Client()
      bucket_name = 'models_ai_save'
      # Construct the blob name using the date and hour
      blob_name = f'model/{current_date}/{current_hour}.joblib'
      # Upload the model to GCS
      bucket = gcs_client.get_bucket(bucket_name)
      blob = bucket.blob(blob_name)
      blob.upload_from_filename('model.joblib')
      print(f"Model uploaded to {blob_name} in GCS.")
    def load_model(self):
        gcs_client = storage.Client()
        bucket_name = 'models_ai_save'
        prefix = 'model/'
        try:
            bucket = gcs_client.get_bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            # Descarga el modelo pre-entrenado si existe
            if blobs:
                # Sort blobs by date and then by hour
                sorted_blobs = sorted(blobs, key=lambda blob: blob.name, reverse=True)
                # Get the most recent blob
                recent_blob = sorted_blobs[0]
                # Download the most recent model
                file_name = 'recent_model.joblib'
                recent_blob.download_to_filename(file_name)
                self.best_model = load(file_name)
                print(f"Loaded model from {recent_blob.name}")
        except Exception as e:
            print(f"Error al cargar el modelo pre-entrenado: {str(e)}")

def load_model():
    if os.path.exists("../key.json"): os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"
    gcs_client = storage.Client()
    bucket_name = 'models_ai_save'
    prefix = 'pipeline/'
    try:
        bucket = gcs_client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        # Descarga el modelo pre-entrenado si existe
        if blobs:
            # Sort blobs by date and then by hour
            sorted_blobs = sorted(blobs, key=lambda blob: blob.name, reverse=True)
            # Get the most recent blob
            recent_blob = sorted_blobs[0]
            # Download the most recent model
            file_name = 'recent_pipe.pkl'
            recent_blob.download_to_filename(file_name)
            print(f"Loaded pipeline from {recent_blob.name}")
            return load(file_name)
        else: return None
    except Exception as e:
        print(f"Error al cargar el pipeline pre-entrenado: {str(e)}")
        return None
