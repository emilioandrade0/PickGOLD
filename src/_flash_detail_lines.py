from playwright.sync_api import sync_playwright
import re

url = 'https://www.flashscore.com.mx/partido/beisbol/houston-astros-lfZXonNq/st-louis-cardinals-IDVz16ES/?mid=prqDeTTF'
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto(url, wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(6000)
    txt = page.inner_text('body')
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    idxs = [i for i,l in enumerate(lines) if re.search(r'\b[1-9]\.\d{2}\b', l)]
    print('matches', len(idxs))
    for i in idxs[:20]:
      print('---')
      for j in range(max(0,i-2), min(len(lines), i+3)):
        print(lines[j])
    b.close()
