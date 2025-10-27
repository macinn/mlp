from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, val_size=0.1, seed=42):
    """
    Dzieli dane na zbiory: treningowy, walidacyjny i testowy.

    test_size — proporcja danych testowych (np. 0.2 oznacza 20%)
    val_size — proporcja danych walidacyjnych w stosunku do treningu (np. 0.1 = 10% z treningowych)
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
