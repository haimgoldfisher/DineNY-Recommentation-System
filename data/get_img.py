from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_image_url_from_google_maps(url):
    default_img = "https://t3.ftcdn.net/jpg/03/24/73/92/360_F_324739203_keeq8udvv0P2h1MLYJ0GLSlTBagoXS48.jpg"
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
        img_elements = driver.find_elements(By.TAG_NAME, 'img')
        if not img_elements:
            return default_img

        for img in img_elements:
            img_src = img.get_attribute('src')
            if img_src and 'https' in img_src and 'googleusercontent' in img_src:
                return img_src

        return default_img
    except Exception as e:
        print(f"Error: {e}")
        return default_img
    finally:
        driver.quit()


def update_image_urls():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Google-Maps-Restaurant']
    restaurants_collection = db['Restaurants']

    restaurants = restaurants_collection.find()
    for restaurant in restaurants:
        gmap_id = restaurant.get('gmap_id')
        url = restaurant.get('url')

        # Check if 'img_url' already exists
        if 'img_url' in restaurant:
            print(f"Skipping gmap_id: {gmap_id} (already has img_url)")
            continue

        # Get image URL
        img_url = get_image_url_from_google_maps(url)
        restaurants_collection.update_one(
            {'gmap_id': gmap_id},
            {'$set': {'img_url': img_url}}
        )
        print(f"Updated image URL for gmap_id: {gmap_id}")


if __name__ == "__main__":
    update_image_urls()
