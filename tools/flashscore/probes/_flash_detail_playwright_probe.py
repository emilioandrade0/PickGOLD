import re
from playwright.sync_api import sync_playwright

url = 'https://www.flashscore.com.mx/partido/beisbol/houston-astros-lfZXonNq/st-louis-cardinals-IDVz16ES/?mid=prqDeTTF'
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto(url, wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(5000)
    txt = page.inner_text('body')
    nums = re.findall(r'\b\d+\.\d{2}\b', txt)
    print('decimal_count', len(nums))
    print('sample', nums[:25])
    # Print lines containing likely odds-like numbers in 1.xx to 6.xx
    for line in txt.splitlines():
        if re.search(r'\b[1-6]\.\d{2}\b', line):
            print('LINE', line[:200])
    b.close()
