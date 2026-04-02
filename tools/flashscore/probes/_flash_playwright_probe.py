from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.flashscore.com.mx/beisbol/', wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(4000)

    # Try several likely selectors used by Flashscore.
    selectors = [
        'div.event__match',
        'div[data-testid="wcl-row"]',
        'div.wcl-row',
        'div.sportName',
        'div.event__wrapper',
        'div[role="row"]',
    ]
    for sel in selectors:
        cnt = page.locator(sel).count()
        print(sel, cnt)

    rows = page.locator('div.event__match')
    print('event_rows', rows.count())
    for i in range(min(8, rows.count())):
        txt = rows.nth(i).inner_text().replace('\n', ' | ')
        print(i, txt[:240])

    browser.close()
