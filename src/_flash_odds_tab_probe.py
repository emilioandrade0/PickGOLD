from playwright.sync_api import sync_playwright
import re

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto('https://www.flashscore.com.mx/beisbol/', wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(3000)
    page.locator('div.filters__tab[data-analytics-alias="odds"]').first.click(timeout=15000)
    page.wait_for_timeout(3000)
    rows = page.locator('div.event__match')
    print('rows', rows.count())
    for i in range(min(10, rows.count())):
        t = rows.nth(i).inner_text().replace('\n',' | ')
        if re.search(r'\b[1-9]\.\d{2}\b', t):
            print(i, t[:280])
    b.close()
