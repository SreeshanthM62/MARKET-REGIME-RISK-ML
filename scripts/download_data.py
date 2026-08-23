from src.data_loader import download_market_data

if __name__ == "__main__":
    download_market_data(
        start="2015-01-01",
        end="2026-01-01",
    )
