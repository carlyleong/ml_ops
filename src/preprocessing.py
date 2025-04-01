import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data
    """
    data = data.drop(columns=["Country"])
    return data

def main():
    """
    Main function
    """
    data = pd.read_csv("data/happy/2015.csv")
    preprocessed_data = preprocess_data(data)
    preprocessed_data.to_csv("data/happy/2015_preprocessed.csv", index=False)

if __name__ == "__main__":
    main()