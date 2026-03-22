from playwright.sync_api import sync_playwright
import re

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.flashscore.com.mx/beisbol/', wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(4000)
    rows = page.locator('div.event__match')
    print('rows', rows.count())
    for i in range(min(6, rows.count())):
        h = rows.nth(i).inner_html()
        if re.search(r'odds|wcl|1\.|2\.', h, re.I):
            print('\nROW', i)
            print(h[:2200])
    # Print classes containing "odds" on page
    classes = page.eval_on_selector_all('*', 'els => Array.from(new Set(els.flatMap(e => (e.className||"" ).toString().split(/\\s+/).filter(c => c.toLowerCase().includes("odds"))))).slice(0,120)')
    print('\nODDS_CLASSES', classes)
    browser.close()
