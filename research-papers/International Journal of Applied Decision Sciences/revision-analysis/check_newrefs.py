import json, urllib.request, urllib.parse, sys
DOIS = ["10.1504/IJADS.2026.153506","10.1504/IJADS.2026.153504",
        "10.1016/j.iimb.2026.100675","10.1038/s41599-026-07653-7",
        "10.1007/s44196-024-00664-9","10.1186/s40537-025-01185-8",
        "10.1057/s41599-026-06661-x","10.1057/s41283-025-00165-9",
        "10.3390/analytics5010009"]
def get(u):
    r = urllib.request.Request(u, headers={'User-Agent':'ref-check/1.0 (mailto:divyamnavin@gmail.com)'})
    return json.load(urllib.request.urlopen(r, timeout=30))
for d in DOIS:
    try:
        m = get('https://api.crossref.org/works/' + urllib.parse.quote(d))['message']
        au = '; '.join((a.get('family','?') + ', ' + a.get('given','?')) for a in m.get('author',[])[:6])
        yr = (m.get('published') or m.get('issued',{})).get('date-parts',[[None]])[0][0]
        print('OK   %s' % d)
        print('     title : %s' % (m.get('title',['?'])[0]))
        print('     journal: %s  vol %s no %s pp %s  (%s)' % (
            (m.get('container-title') or ['?'])[0], m.get('volume','-'),
            m.get('issue','-'), m.get('page','-'), yr))
        print('     authors: %s' % au)
    except Exception as e:
        print('FAIL %s  -> %s' % (d, e))
    print()
