import re
import requests

url = "https://www.flashscore.com.mx/beisbol/"
html = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
print("has_nuxt", "__NUXT__" in html)
print("has_next", "__NEXT_DATA__" in html)
srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
print("script_src_count", len(srcs))
for s in srcs[:30]:
    print(s)
print("--- odds context ---")
m = re.search(r"odds", html, re.I)
if m:
    st = max(0, m.start() - 160)
    en = min(len(html), m.start() + 320)
    print(html[st:en].replace("\n", " "))
else:
    print("no odds token")
