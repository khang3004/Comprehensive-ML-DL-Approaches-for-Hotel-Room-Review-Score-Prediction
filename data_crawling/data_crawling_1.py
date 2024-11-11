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


for i in range(6):
    pulled_data = crawling_from_booking_2(i)
    print(pulled_data[0])