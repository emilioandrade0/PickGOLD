import re
import requests
html = requests.get('https://www.flashscore.com.mx/beisbol/', timeout=20, headers={'User-Agent':'Mozilla/5.0'}).text
for pat in [r'/x/feed/[A-Za-z0-9_\-]+', r'd\.flashscore\.com', r'/res/_fs/', r'event_id', r'bookmaker', r'odds_enable', r'livetable', r'window\.[A-Za-z_]+\s*=']:
    ms = re.findall(pat, html)
    print(pat, len(ms))
print('contains_x_feed', '/x/feed/' in html)
