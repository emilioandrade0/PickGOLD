import re
import requests
from urllib.parse import urljoin

base = "https://www.flashscore.com.mx"
scripts = [
    "/res/_fs/build/mainPageScripts.3729163.js",
    "/res/_fs/build/liveTable.13b11b7.js",
    "/x/js/core_201_2296000000.js",
]
patterns = [r"/x/feed/[a-zA-Z0-9_\-]+", r"odds", r"bookmaker", r"participants", r"event_", r"XHR", r"fetch\("]
for s in scripts:
    u = urljoin(base, s)
    try:
        txt = requests.get(u, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
    except Exception as e:
        print('ERR', s, e)
        continue
    print('\n===', s, 'len', len(txt))
    for p in patterns:
        found = re.findall(p, txt)
        print(p, 'count', len(found))
    feeds = sorted(set(re.findall(r"/x/feed/[a-zA-Z0-9_\-]+", txt)))
    if feeds:
        print('sample_feeds', feeds[:20])
