from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle API initialisieren
api = KaggleApi()
api.authenticate()

# Datensatz herunterladen
api.dataset_download_files('meowmeowmeowmeowmeow/gtsrb-german-traffic-sign', path='.', unzip=False)

# ZIP-Datei entpacken
with ZipFile('gtsrb-german-traffic-sign.zip', 'r') as zip_ref:
    zip_ref.extractall('gtsrb-german-traffic-sign')