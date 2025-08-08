import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

pdb_ids = []
pdb_file_list = sorted(os.listdir(""))
for fasta_name in pdb_file_list:
    pdb_ids.append(fasta_name[:4])

download_path = "./download"
url = 'http://rnapdbee.cs.put.poznan.pl/'

chrome_options = Options()
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--allow-running-insecure-content")
chrome_options.add_argument(f"--unsafely-treat-insecure-origin-as-secure={url}")
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_path,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

driver_path = r""
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)




for pdb_id in pdb_ids:

    driver.get(url)


    WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.ID, "pdbId"))
    )

    input_field = driver.find_element(By.ID, "pdbId")
    input_field.clear()  # 清空输入框
    input_field.send_keys(pdb_id)

    get_button = driver.find_element(By.XPATH, "//input[@type='button' and @value='Get']")
    get_button.click()


    WebDriverWait(driver, 40).until(
        EC.element_to_be_clickable((By.ID, "commitPdb"))
    )


    run_button = driver.find_element(By.ID, "commitPdb")
    run_button.click()

    time.sleep(5)

    select_all_button = driver.find_element(By.ID, "selectClearAllTop")
    select_all_button.click()

    time.sleep(2)

    download_button = driver.find_element(By.ID, "downloadAllTop")
    download_button.click()


    time.sleep(10)

    print(f"Download complete for PDB ID: {pdb_id}")

time.sleep(70)
driver.quit()


