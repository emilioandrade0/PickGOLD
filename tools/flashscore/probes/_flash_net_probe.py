from playwright.sync_api import sync_playwright

url = 'https://www.flashscore.com.mx/partido/beisbol/houston-astros-lfZXonNq/st-louis-cardinals-IDVz16ES/?mid=prqDeTTF'
seen = []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()

    def on_response(resp):
        u = resp.url
        if any(k in u.lower() for k in ['odds', '/x/feed/', 'bookmaker', 'event', 'match']):
            ct = (resp.headers.get('content-type') or '').lower()
            seen.append((u, ct, resp.status))

    page.on('response', on_response)
    page.goto(url, wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(6000)
    b.close()

# unique + print
uniq = []
for item in seen:
    if item not in uniq:
        uniq.append(item)
print('captured', len(uniq))
for u,ct,st in uniq[:120]:
    print(st, ct, u)
