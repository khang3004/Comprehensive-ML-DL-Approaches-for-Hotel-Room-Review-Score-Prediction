from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import time
import numpy as np
from bs4 import BeautifulSoup
import requests
import pandas as pd
from urllib.parse import urlparse
import warnings
warnings.simplefilter('ignore')
from utils_to_crawl import *


properties=load_properties(5)
print(properties['link'])

for i in range(5):
    properties=load_properties(i)
    test = crawling_from_links(properties,i)
    print(test['review_score'])

full_data = pd.DataFrame()  

for i in range(6):
    data = pd.read_excel(f'cleaning_data_{i}.xlsx')
    data['label'] = i
    full_data = pd.concat([full_data, data], ignore_index=True)

full_data['subscore'] = full_data['subscore'].apply(eval)
df = pd.DataFrame.from_records(full_data['subscore'].tolist())
full_data = pd.concat([full_data, df], axis=1)

full_data = full_data.rename(columns={'Nhân viên phục vụ': 'staff_score', 'Tiện nghi': 'facility_score',
                                    'Sạch sẽ': 'clean_score', 'Thoải mái': 'Comfor_score',
                                    'Đáng giá tiền': 'price_score', 'Địa điểm': 'place_score',
                                    'WiFi miễn phí': 'wifi_score'})

full_data['price'] = full_data['price'].apply(convert_price, convert_dtype=False, convert_missing=False)
list_col = ['review_score','staff_score', 'facility_score', 'clean_score', 'Comfor_score', 'price_score', 'place_score', 'wifi_score']
for name in list_col:
# Create a scatter plot of Price vs. Review Count
    full_data[name] = full_data[name].astype(str).str.replace(',', '.')
    full_data[name] = pd.to_numeric(full_data[name], errors='coerce')
full_data = full_data.drop(columns=['subscore', 'facilities', 'Unnamed: 0'])

full_data.to_csv('fulldata.csv')

