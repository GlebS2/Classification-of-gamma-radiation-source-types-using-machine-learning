from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def handle_outliers(df: pd.DataFrame, 
                    names: Union[str, list, None] = None, 
                    low_quantile: Union[float, list] = 0, 
                    up_quantile: Union[float, list] = 1, 
                    method: str = 'trunc_outliers'):
    """
    Удаляет строки с выбросами (или наоборот выделяет их) в DataFrame по данным из заданных столбцов по заданным квантилям.
    
    Если указано несколько столбцов, то выбросом считается строка с аномальным значением в любом из них.
    Если names=None, выбираются все столбцы DataFrame для обработки выбросов.

    Аргументы:
    - df: pandas DataFrame, входные данные
    - names: str или list of str, имена столбцов, в которых производится обработка выбросов
    - low_quantile: float или list of float, нижние квантили для отсечения выбросов
    - up_quantile: float или list of float, верхние квантили для отсечения выбросов
    - method: str, метод обработки выбросов ('trunc_outliers' для удаления выбросов или 'show_outliers' для сохранения только выбросов)

    Возвращает:
    - pd.DataFrame: DataFrame без выбросов или DataFrame с выбросами в зависимости от выбранного метода

    Пример использования:
    handle_outliers(df, 'column_name', low_quantile=0.05, up_quantile=0.95, method='trunc_outliers')
    """
    
    # Если names=None, выбираем все столбцы DataFrame для обработки выбросов
    if names is None:
        names = df.columns.tolist()
    
    # Преобразование одиночной строки в массив
    if isinstance(names, str):
        names = [names]
    
    # Проверка, не являются ли low_quantile и up_quantile списком
    if not isinstance(low_quantile, (list, tuple)):
        low_quantile = [low_quantile] * len(names)
    if not isinstance(up_quantile, (list, tuple)):
        up_quantile = [up_quantile] * len(names)
    
    # Проверка соответствия числа квантилей числу столбцов
    if len(low_quantile) != len(up_quantile) != len(names):
        raise ValueError("Количество квантилей не соответствует количеству столбцов")
    
    # Определение метода для работы с выбросами
    if method == 'trunc_outliers':
        for col_name, low_q, up_q in zip(names, low_quantile, up_quantile):
            lower_bound = df[col_name].quantile(low_q)
            upper_bound = df[col_name].quantile(up_q)
            df = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]
            
    elif method == 'show_outliers':
        outliers = pd.DataFrame()
        for col_name, low_q, up_q in zip(names, low_quantile, up_quantile):
            lower_bound = df[col_name].quantile(low_q)
            upper_bound = df[col_name].quantile(up_q)
            outliers = pd.concat([outliers, df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]])
        return outliers
    
    return df

def normalize_source_type(data):
    """
    Меняет обозначения в столбце 'source_type', различая лишь квазары, пульсары, 'unknown' и 'other' типы.

    Аргументы:
    - data: pandas DataFrame, входные данные

    Возвращает:
    - pandas DataFrame: DataFrame с замененными значениями столбца 'source_type'

    Пример использования:
    normalized_data = normalize_source_type(data)
    """
    from import_dat import replacements

    # Создаем копию данных, чтобы избежать изменения исходного DataFrame
    df = data.copy()

    # Заменяем значения с использованием словаря replacements
    df['source_type'].replace(replacements, inplace=True)

    # Заменяем остальные значения на 'other'
    df['source_type'].replace(to_replace='^(?!unknown|blazar|pulsar).*$', value='other', regex=True, inplace=True)

    return df


def visualize_training_history(history, figsize=(12, 4)):
    """
    Визуализирует историю обучения по метрикам потерь и точности.

    Аргументы:
    - history: История обучения модели Keras, передаваемая в конце работы метода model.fit() 
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)  # Создаем подграфики 1x2

    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)  # Включаем сетку

    axs[1].plot(history.history['accuracy'], label='Training Accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)  # Включаем сетку

    plt.tight_layout()  # Для правильного размещения графиков
    plt.show()
    
    
def combine_predictions(predictions, threshold=0.8, ratio_threshold=2.):
    """
    Из предсказаний нейросети делает вывод о принадлежности образца определённому классу.

    Аргументы:
    - predictions: массив или список массивов с предсказаниями модели
    - threshold: пороговое значение для принятия решения о классе
    - ratio_threshold: пороговое значение для определения "значительного" превышения предсказания

    Возвращает:
    - массив, содержащий объединенные предсказанные классы
    """
    combined = np.empty(len(predictions), dtype=object)

    for i, pred in enumerate(predictions):
        max_pred = np.max(pred)
        max_index = np.argmax(pred)
        other_predictions = np.delete(pred, max_index)

        if max_pred >= threshold and max_pred > ratio_threshold * np.max(other_predictions):
            combined[i] = max_index
        else:
            combined[i] = 'None'

    return combined


def augmentation(data, error_columns, times=1):
    """
    Функция для аугментации данных.

    Параметры:
    - data: pandas.DataFrame
        Исходный датафрейм с данными.
    - error_columns: list
        Список колонок, к которым будет добавлен шум.
    - times: int, optional
        Количество раз, которое нужно аугментировать данные (по умолчанию 1).

    Возвращает:
    - data: pandas.DataFrame
        Аугментированный датафрейм с данными.
    """
    
    # Копируем датафреймы
    data_aug = data.copy()
    
    # Определяем колонки, к которым будем добавлять шум
    columns = [err_col.rstrip('_error') for err_col in error_columns]
    
    for i in range(times):
        
        # Аугментируем данные
        aug = data_aug.copy()
        stds = aug[error_columns].to_numpy()
        aug[columns] += np.random.normal(0, stds)

        # Конкатенируем датафреймы
        data = pd.concat([data, aug])
        
    return data


def normalize_df(df, scaler):
    """
    Нормализует числовые столбцы в DataFrame с использованием заданного скейлера.

    Аргументы:
    df (pandas.DataFrame): Исходный DataFrame.
    scaler: Объект скейлера, используемый для нормализации данных.

    Возвращает:
    pandas.DataFrame: Нормализованный DataFrame с сохранением нечисловых столбцов.

    """
    numeric_columns = df.select_dtypes(include='number')
    non_numeric_columns = df.select_dtypes(exclude='number')
    scaled_data = scaler.transform(df.select_dtypes(include='number'))
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns.columns, index=numeric_columns.index)
    
    return non_numeric_columns.join(scaled_df)