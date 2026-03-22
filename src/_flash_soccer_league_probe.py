from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto('https://www.flashscore.com.mx/futbol/', wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(3000)
    page.locator('div.filters__tab[data-analytics-alias="odds"]').first.click(timeout=15000)
    page.wait_for_timeout(3000)
    rows = page.locator('div.event__match[data-event-row="true"]')
    print('rows', rows.count())
    for i in range(min(15, rows.count())):
        row = rows.nth(i)
        home = row.locator('.event__participant--home').first.inner_text().strip() if row.locator('.event__participant--home').count() else ''
        if not home:
            continue
        league_loc = row.locator("xpath=preceding-sibling::div[contains(@class,'headerLeague__wrapper')][1]//span[contains(@class,'headerLeague__title-text')]").first
        cat_loc = row.locator("xpath=preceding-sibling::div[contains(@class,'headerLeague__wrapper')][1]//span[contains(@class,'headerLeague__category-text')]").first
        league = league_loc.inner_text().strip() if league_loc.count() else ''
        cat = cat_loc.inner_text().strip() if cat_loc.count() else ''
        print(i, 'CAT=', cat, 'LEAGUE=', league, 'HOME=', home)
    b.close()
