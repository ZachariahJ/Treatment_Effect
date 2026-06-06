import pandas as pd

def data_preprocessing(csv_path: str = "data_generated.csv") -> pd.DataFrame:
    # Read file as Data Frame
    df: pd.DataFrame = pd.read_csv(csv_path)

    # Imputation for missing values
    # Set numerical values to 0
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Map Categorical embeddings to Dim 8 per column
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes


    # Standardization for numerical columns
    for col in num_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    


    return df


if __name__ == "__main__":
    # Print the first 5 rows of the Data Frame
    df = data_preprocessing()
    # print(df.head())
    # print(df.shape)              # (行数, 列数)
    # print(df.head())             # 看前 5 行长什么样
    print(df.dtypes)             # 每列的数据类型（数字 or 文字）
    # print(df.isna().sum())       # 每列有多少缺失值
    # print(df.describe())         # 数值列的统计：均值、最小、最大、分位数
    # print(df["GENDER"].value_counts())   # 某个类别列各取值出现多少次