import requests
from datetime import datetime

def download_bhavcopy(date, save_dir="data/bhavcopy/"):
    date_str = date.strftime("%d%b%Y").upper()
    month_str = date.strftime("%b").upper()
    year_str = date.strftime("%Y")
    url = f"https://archives.nseindia.com/content/historical/EQUITIES/{year_str}/{month_str}/cm{date_str}bhav.csv.zip"
    file_name = f"{save_dir}cm{date_str}bhav.csv.zip"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            return file_name
        else:
            return None
    except Exception as e:
        print(f"Download error: {e}")
        return None

