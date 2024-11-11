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
from time import sleep
import warnings
warnings.simplefilter('ignore')
from fix_functions import *
def safe_extract1(element, query, attr=None, default=np.nan):
    try:
        result = element.find(query).get(attr) if attr else element.find(query).text
        return result.strip() if result else default
    except AttributeError:
        return default




def get_full_url_path(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    path = parsed_url.path
    full_path = f"{scheme}://{netloc}{path}"
    return full_path

def extract_hotel_links(url, processed_links):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    property_cards = soup.find_all("div", {"data-testid": "property-card"}) 
    print(len(property_cards))    
    properties = []
    for property in property_cards:
        new_property = {}
        try:
            link = property.find('a', {'data-testid': 'title-link'}).get('href')
            linked = fix_link(link)
            new_property['link'] = get_full_url_path(linked)
        except:
            new_property['link'] = np.nan
        # Skip if duplicate
        if new_property['link'] in processed_links:
            continue
        try:
            new_property['hotel_name'] = property.find('div', {'data-testid': 'title'}).text
        except:
            new_property['hotel_name'] = np.nan
        try:
            review_score, review_count = property.find('div', {'data-testid': 'review-score'})
            new_property['review_score'] = review_score.text.split(" ")[2]
            
            new_property['review_count'] = review_count.text.split(" ")[-3]
        except:
            new_property['review_score'] = np.nan
            new_property['review_count'] = np.nan
        try:
            new_property['price'] = property.find('span', {'data-testid': 'price-and-discounted-price'}).text
        except:
            new_property['price'] = np.nan
        try:
            new_property['address'] = property.find('span', {'data-testid': 'address'}).text
        except:
            new_property['address'] = np.nan
        try:
            new_property['image'] = property.find('img', {'data-testid': 'image'}).get('src')
        except:
            new_property['image'] = np.nan
        properties.append(new_property)
        processed_links.append(new_property['link'])
    return properties, processed_links
# def extract_hotel_links(url, processed_links):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    
    # Gửi request và parse HTML
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    property_cards = soup.find_all("div", {"data-testid": "property-card"})
    print(f"Found {len(property_cards)} property cards.")

    # Tạo danh sách lưu trữ properties
    properties = []
    # Duyệt qua từng thẻ khách sạn
    for property in property_cards:
        link = safe_extract1(property, 'a[data-testid="title-link"]', 'href')
        full_link = fix_link(link) if link else np.nan
        if full_link in processed_links:
            continue

        # Thu thập thông tin khách sạn
        new_property = {
            'link': get_full_url_path(full_link),
            'hotel_name': safe_extract(property, 'div[data-testid="title"]'),
            'review_score': safe_extract(property, 'div[data-testid="review-score"]', default=np.nan),
            'review_count': safe_extract(property, 'div[data-testid="review-score"]', default=np.nan),
            'price': safe_extract(property, 'span[data-testid="price-and-discounted-price"]'),
            'address': safe_extract(property, 'span[data-testid="address"]'),
            'image': safe_extract(property, 'img[data-testid="image"]', 'src')
        }

        # Thêm vào danh sách properties và processed_links
        properties.append(new_property)
        processed_links.append(new_property['link'])

    return properties, processed_links


def init_driver(url):
    """Khởi tạo trình duyệt và mở trang web."""
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    return driver

def scroll_to_load(driver, wait_time=2):
    """Cuộn trang để tải thêm dữ liệu."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(wait_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def get_last_page(driver):
    """Lấy số trang cuối cùng từ kết quả tìm kiếm."""
    try:
        pages = driver.find_elements(By.XPATH, '//div[@data-testid="pagination"]//li')
        return int(pages[-1].text) if pages else 1
    except (ValueError, IndexError):
        return 1

# def crawling_from_booking(filter_star,check_in_day = '2024-10-15', check_out_day = '2024-10-21'):
    url = f"https://www.booking.com/searchresults.vi.html?aid=304142&ss=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne_untouched=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&lang=vi&dest_id=-3730078&dest_type=city&checkin={check_in_day}&checkout={check_out_day}&group_adults=2&no_rooms=1&group_children=0&nflt=class%3D{filter_star}&order=price&offset="
    
    # Khởi tạo trình duyệt và truy cập URL
    driver = init_driver(url + "25")
    processed_links = []
    properties = []

    # Xác định trang cuối cùng
    last_page = get_last_page(driver)

    # Duyệt qua từng trang
    for current_page in range(last_page):
        driver.get(url + str(current_page * 25))
        scroll_to_load(driver)  # Cuộn trang để tải thêm dữ liệu
        print(f"Processing: {driver.current_url}")
        
        # Lấy thông tin khách sạn và cập nhật danh sách
        new_properties, processed_links = extract_hotel_links(driver.current_url, processed_links)
        properties.extend(new_properties)

    # Lưu kết quả vào file Excel
    pd.DataFrame(properties).to_excel(f"Crawling_results_{filter_star}.xlsx")
    return properties


def crawling_from_booking_2(filter_star):
    url = f"https://www.booking.com/searchresults.vi.html?aid=304142&ss=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne_untouched=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&highlighted_hotels=10322077&efdco=1&lang=vi&dest_id=-3730078&dest_type=city&checkin=2024-04-10&checkout=2024-04-11&group_adults=2&no_rooms=1&group_children=0&order=class&nflt=class%3D{filter_star}"
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    processed_links = []
    properties = []
    # Lấy chiều cao ban đầu của trang
    for i in range(10):
        driver.execute_script("window.scrollBy(0, 2000);")
        time.sleep(1)  # Đợi để trang web tải thêm dữ liệu
    # Lấy tất cả kết quả đã tải
    # ád
    new_properties, processed_links = extract_hotel_links(driver.current_url, processed_links)
    properties.extend(new_properties)

    df = pd.DataFrame(properties)
    df.to_excel(f"Scrolled_crawling_{filter_star}.xlsx")
    return properties

# def crawling_from_links(data,filter_star):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
    properties = data.copy()
    properties['facilities'] =np.full(len(properties), np.nan)
    properties['subscore'] = np.full(len(properties), np.nan)
    for item in range(len(properties)):
        properties['link'][item] = get_full_url_path(properties['link'][item])
        properties['address'][item] = fix_address(properties['address'][item])
        try:
            properties['review_score'][item] = fix_scores(properties['review_score'][item])
        except:
            properties['review_score'][item] = np.nan
        resp = requests.get(properties['link'][item], headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')
        try:
            facilities = soup.find('div',{'data-testid':'property-most-popular-facilities-wrapper'}).text
            properties['facilities'][item] = fix_facilities(facilities)
        except:
            properties['facilities'][item] = np.nan
        try:
            subscore = soup.find_all('div',{'data-testid':'review-subscore'})
            review_subscore = [sub.get_text(strip=True) for sub in subscore]
            properties['subscore'][item] = fix_subscore(review_subscore)
        except:
            properties['subscore'][item] = np.nan
    properties.to_excel(f"new_crawling_{filter_star}.xlsx")
    return properties

def crawling_from_booking(filter_star, check_in_day = 2024-10-19, check_out_day = 2024-10-23):
    url = f"https://www.booking.com/searchresults.vi.html?aid=304142&ss=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne_untouched=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&lang=vi&dest_id=-3730078&dest_type=city&checkin={check_in_day}&checkout={check_out_day}&group_adults=2&no_rooms=1&group_children=0&nflt=class%3D{filter_star}&order=price&offset="
    # url = f"https://www.booking.com/searchresults.vi.html?laid=397594&ss=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&ssne_untouched=TP.+H%C3%B4%CC%80+Chi%CC%81+Minh&lang=vi&dest_id=-3730078&dest_type=city&checkin=2024-04-02&checkout=2024-04-03&group_adults=2&no_rooms=1&group_children=0&nflt=class%3D{filter_star}&order=price&offset="
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url+"25")
    processed_links = []
    properties = []
    try:
        pages = driver.find_elements(By.XPATH, '//div[@data-testid="pagination"]//li')
        last_page = int(pages[-1].text)
    except:
        last_page = 1
    for current_page in range(last_page):
            driver.get(url+str(current_page*25))
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2) # Đợi để trang web tải thêm kết quả
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            time.sleep(5)
            print(driver.current_url)
            new_properties, processed_links = extract_hotel_links(driver.current_url, processed_links)
            properties = properties + new_properties
    df = pd.DataFrame(properties)
    df.to_excel(f"special_raw_data_{filter_star}.xlsx")
    return properties

def fetch_and_parse(url, headers):
    """Gửi request đến URL và parse HTML."""
    resp = requests.get(url, headers=headers)
    return BeautifulSoup(resp.text, 'html.parser')

def safe_extract(soup, query, multiple=False):
    """Trích xuất văn bản hoặc danh sách văn bản từ HTML một cách an toàn."""
    try:
        if multiple:
            return [element.get_text(strip=True) for element in soup.find_all(query)]
        return soup.find(query).text.strip()
    except AttributeError:
        return np.nan

def crawling_from_links(data, filter_star):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }

    properties = data.copy()
    properties['facilities'] = np.nan
    properties['subscore'] = np.nan

    # Duyệt qua từng khách sạn và crawl thông tin chi tiết
    for i, row in properties.iterrows():
        properties.at[i, 'link'] = get_full_url_path(row['link'])
        properties.at[i, 'address'] = fix_address(row['address'])
        properties.at[i, 'review_score'] = fix_scores(row.get('review_score', np.nan))

        # Gửi request và parse HTML
        soup = fetch_and_parse(properties.at[i, 'link'], headers)

        # Lấy thông tin cơ sở vật chất và điểm phụ
        properties.at[i, 'facilities'] = fix_facilities(
            safe_extract(soup, 'div[data-testid="property-most-popular-facilities-wrapper"]')
        )
        properties.at[i, 'subscore'] = fix_subscore(
            safe_extract(soup, 'div[data-testid="review-subscore"]', multiple=True)
        )

    # Lưu kết quả vào file Excel
    properties.to_excel(f"new_crawling_{filter_star}.xlsx", index=False)
    return properties


def load_properties(filter_star):
    with open(f'new_crawling_{filter_star}.xlsx', 'rb') as file:
        properties = pd.read_excel(file)
    return properties


def convert_price(price_str, convert_missing=False):
    if pd.isna(price_str) and not convert_missing:
        return price_str  # Giữ nguyên giá trị NaN
    else:
        price_str = str(price_str)
        price_str = re.sub(r'[^\d,]+', '', price_str)  # Loại bỏ tất cả ký tự không phải số và dấu phẩy
        price_str = price_str.replace(',', '')
        return int(price_str)
    
