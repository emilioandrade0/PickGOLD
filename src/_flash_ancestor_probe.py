from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto('https://www.flashscore.com.mx/beisbol/', wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(5000)

    target = page.locator('div.event__participant--home', has_text='Houston Astros').first
    print('target_count', target.count())
    if target.count():
        # Climb a few ancestors and print snippets
        handles = target.evaluate_handle('el => { const out=[]; let n=el; for(let i=0;i<6 && n;i++){ out.push(n.outerHTML.slice(0,3500)); n=n.parentElement; } return out; }')
        arr = handles.json_value()
        for i,html in enumerate(arr):
            print('\n=== ancestor', i, '===')
            print(html)

    b.close()
