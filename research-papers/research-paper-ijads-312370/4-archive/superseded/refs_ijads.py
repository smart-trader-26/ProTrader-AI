"""Resolve every candidate reference against CrossRef and emit Inderscience bibitems.

Nothing is written from memory: authors, container, volume, issue, pages, year and
DOI all come back from the CrossRef REST API. Also queries CrossRef for recent
(2025-2026) journal articles on the paper's topics so real ones can be selected.
"""
from __future__ import annotations
import io, json, sys, time, urllib.parse, urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
API = 'https://api.crossref.org/works'
MAILTO = 'ijads-refcheck@example.org'
BAD = ('ssrn electronic journal', 'ssrn', 'cfa digest', 'social science research network')
GOOD = ('journal-article', 'proceedings-article', 'book', 'book-chapter', 'monograph')


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': f'ref-check/1.0 (mailto:{MAILTO})'})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def search(title, rows=5):
    q = urllib.parse.urlencode({'query.bibliographic': title, 'rows': rows, 'mailto': MAILTO})
    try:
        return get(f'{API}?{q}')['message']['items']
    except Exception as e:
        print('  ! ', e)
        return []


def recent(topic, rows=12, frm='2025-01-01'):
    q = urllib.parse.urlencode({'query.bibliographic': topic, 'rows': rows, 'mailto': MAILTO,
                                'filter': f'from-pub-date:{frm},type:journal-article',
                                'sort': 'relevance'})
    try:
        return get(f'{API}?{q}')['message']['items']
    except Exception as e:
        print('  ! ', e)
        return []


def year(it):
    for k in ('published-print', 'published-online', 'issued', 'created'):
        if k in it and it[k].get('date-parts', [[None]])[0][0]:
            return it[k]['date-parts'][0][0]
    return None


def fmt(it):
    au = it.get('author', []) or []
    names = []
    for a in au:
        fam = a.get('family', '')
        gv = a.get('given', '')
        ini = ''.join(p[0] + '.' for p in gv.replace('-', ' ').split() if p)
        names.append(f'{fam}, {ini}' if fam else a.get('name', ''))
    cont = (it.get('container-title') or [''])[0]
    return dict(title=(it.get('title') or [''])[0], authors=names, container=cont,
                volume=it.get('volume', ''), issue=it.get('issue', ''),
                page=it.get('page', ''), year=year(it), doi=it.get('DOI', ''),
                type=it.get('type', ''))


WANTED = [
    'Portfolio Selection Markowitz Journal of Finance 1952',
    'A well-conditioned estimator for large-dimensional covariance matrices Ledoit Wolf',
    'A simple positive semi-definite heteroskedasticity and autocorrelation consistent covariance matrix Newey West',
    'The stationary bootstrap Politis Romano',
    'A new approach to the economic analysis of nonstationary time series and the business cycle Hamilton',
    'International asset allocation with regime shifts Ang Bekaert',
    'Optimal versus naive diversification how inefficient is the 1/N portfolio strategy DeMiguel Garlappi Uppal',
    'Volatility-managed portfolios Moreira Muir',
    'Building diversified portfolios that outperform out of sample Lopez de Prado',
    'When is a liability not a liability Textual analysis dictionaries and 10-Ks Loughran McDonald',
    'The deflated Sharpe ratio Bailey Lopez de Prado',
    'Backtesting Harvey Liu Journal of Portfolio Management',
    'Robust performance hypothesis testing with the Sharpe ratio Ledoit Wolf',
    'Regime shifts implications for dynamic strategies Kritzman Page Turkington',
    'Risk everywhere modeling and managing volatility Bollerslev Hood Huss Pedersen',
    'Momentum has its moments Barroso Santa-Clara',
    'Empirical asset pricing via machine learning Gu Kelly Xiu',
    'Giving content to investor sentiment the role of media in the stock market Tetlock',
]

TOPICS = [
    'regime switching asset allocation portfolio',
    'machine learning portfolio construction optimization',
    'deep reinforcement learning portfolio allocation',
    'market regime detection clustering financial markets',
    'financial sentiment analysis large language models stock',
    'volatility targeting risk budgeting portfolio',
    'covariance shrinkage estimation portfolio optimization',
    'backtest overfitting evaluation trading strategies',
    'multimodal fusion stock prediction',
    'decision making under uncertainty investment portfolio',
]

print('=' * 100)
print('RESOLVED CANDIDATES')
print('=' * 100)
# CrossRef's best title match is not always the version of record; pin these by DOI.
BY_DOI = {'Optimal versus naive diversification how inefficient is the 1/N portfolio strategy DeMiguel Garlappi Uppal': '10.1093/rfs/hhm075',
          'The deflated Sharpe ratio Bailey Lopez de Prado': '10.3905/jpm.2014.40.5.094'}

out = []
for w in WANTED:
    if w in BY_DOI:
        try:
            its = [get(f'{API}/{urllib.parse.quote(BY_DOI[w])}')['message']]
        except Exception as e:
            print('  ! DOI', BY_DOI[w], e); its = []
    else:
        its = [i for i in search(w) if i.get('type') in GOOD
               and (i.get('container-title') or [''])[0].lower() not in BAD]
    if not its:
        print(f'\n?? NOT FOUND: {w}')
        continue
    r = fmt(its[0])
    out.append(r)
    print(f"\n{r['year']}  {r['container']}")
    print(f"   {', '.join(r['authors'][:6])}")
    print(f"   {r['title']}")
    print(f"   vol {r['volume']} no {r['issue']} pp {r['page']}  doi {r['doi']}")
    time.sleep(0.3)

print('\n' + '=' * 100)
print('RECENT (2025+) CANDIDATES BY TOPIC')
print('=' * 100)
rec = {}
for t in TOPICS:
    print(f'\n--- {t}')
    seen = []
    for i in recent(t):
        r = fmt(i)
        if not r['year'] or r['year'] < 2025 or r['container'].lower() in BAD:
            continue
        if not r['container'] or not r['authors']:
            continue
        seen.append(r)
        print(f"   {r['year']} | {r['container'][:52]:52s} | {r['title'][:72]}")
        print(f"        {', '.join(r['authors'][:4])} | vol {r['volume']} no {r['issue']} "
              f"pp {r['page']} | {r['doi']}")
    rec[t] = seen
    time.sleep(0.3)

json.dump(dict(wanted=out, recent=rec), open('refs_candidates.json', 'w'), indent=1)
print('\nsaved refs_candidates.json')
