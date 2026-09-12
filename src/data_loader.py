import pandas as pd
from src.config import TICKER, START_DATE, END_DATE


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [column[0] for column in df.columns]

    return df


def download_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df.reset_index(inplace=True)
    df = _flatten_yfinance_columns(df)
    if "Date" not in df.columns or df.empty:
        return df
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def save_raw_data(df):
    df.to_csv("data/raw/stock_data.csv", index=False)

if __name__ == "__main__":
    df = download_data()
    save_raw_data(df)
    print(df.head())

    
