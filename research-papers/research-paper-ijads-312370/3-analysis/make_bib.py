"""Emit the Inderscience bibliography for ai67.tex from CrossRef records only.

Every entry is resolved by DOI; authors, title, container, volume, issue, pages and
year come back from the API, never from memory. Enforces the journal's limits of at
most three references per journal and at most three per author.
"""
from __future__ import annotations
import html, io, json, re, sys, time, urllib.parse, urllib.request
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
API = 'https://api.crossref.org/works'
MAILTO = 'ijads-refcheck@example.org'

# key -> DOI. Order here is the order in the bibliography.
ENTRIES = [
    ('markowitz52', '10.2307/2975974'),
    ('hamilton89', '10.2307/1912559'),
    ('newey87', '10.2307/1913610'),
    ('politis94', '10.1080/01621459.1994.10476870'),
    ('angbekaert02', '10.1093/rfs/15.4.1137'),
    ('ledoit04', '10.1016/s0047-259x(03)00096-4'),
    ('ledoit08', '10.1016/j.jempfin.2008.03.002'),
    ('demiguel09', '10.1093/rfs/hhm075'),
    ('loughran11', '10.1111/j.1540-6261.2010.01625.x'),
    ('kritzman12', '10.2469/faj.v68.n3.3'),
    ('barroso15', '10.1016/j.jfineco.2014.11.010'),
    ('harvey15', '10.3905/jpm.2015.42.1.013'),
    ('prado16', '10.3905/jpm.2016.42.4.059'),
    ('moreira17', '10.1111/jofi.12513'),
    ('gu20', '10.1093/rfs/hhaa009'),
    # retained IEEE material from the submitted version
    ('lee07', '10.1109/TSMCA.2007.904825'),
    ('chou18', '10.1109/TII.2018.2794389'),
    ('huang20', '10.1109/TFUZZ.2019.2904920'),
    ('zhang22', '10.1109/ACCESS.2022.3195942'),
    ('farimani24', '10.1109/ACCESS.2024.3441029'),
    ('sami25', '10.1109/ACCESS.2025.3543741'),
    # recent journal literature, 2025-2026
    ('li25', '10.3390/math13172837'),
    ('jiang25', '10.1016/j.iref.2025.103996'),
    ('ashrafzadeh25', '10.1016/j.rineng.2025.106263'),
    ('asimit25', '10.1016/j.insmatheco.2025.103139'),
    ('hood25', '10.3905/jpm.2025.1.764'),
    ('agal25', '10.1038/s41598-025-26337-x'),
    ('lu25', '10.3390/e27101083'),
    ('mun25', '10.3390/jtaer20020077'),
    ('zong25', '10.1007/s40747-025-02023-3'),
    ('chuang25', '10.3390/bdcc9100263'),
    ('luo26', '10.3905/jfds.2026.013'),
    ('ha26', '10.1186/s40854-026-00927-8'),
    ('wang26', '10.1186/s40854-026-00929-6'),
    ('saijai26', '10.1108/ajeb-10-2025-0168'),
    ('officioso26', '10.1016/j.ins.2026.123711'),
    ('muhammad26', '10.1007/s41060-026-01201-x'),
    ('raimundo26', '10.1057/s41599-026-07252-6'),
    ('valls26', '10.4236/jfrm.2026.153012'),
    ('kehinde25', '10.3390/ijfs13040192'),
    ('sheppert26', '10.3390/jrfm19010060'),
    ('singsiri26', '10.3390/ijfs14050112'),
    # Int. J. of Applied Decision Sciences, as the editor requested
    ('khodamoradi23', '10.1504/ijads.2023.129475'),
    ('zhangz25', '10.1504/ijads.2025.143079'),
    ('mazzotta25', '10.1504/ijads.2025.10066241'),
]

# Books are not reliably in CrossRef; entered by hand from the printed edition.
MANUAL = [
    ('grinold00', 'Grinold, R.C. and Kahn, R.N.', 'Grinold and Kahn', '2000',
     "Grinold, R.C. and Kahn, R.N. (2000) {\\it Active Portfolio Management: A Quantitative "
     "Approach for Producing Superior Returns and Controlling Risk}, 2nd ed., "
     "McGraw-Hill, New York, NY."),
]


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': f'ref-check/1.0 (mailto:{MAILTO})'})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def clean(s):
    s = re.sub(r'<[^>]+>', '', s or '')
    s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()


def tex(s):
    for a, b in [('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_')]:
        s = s.replace(a, b)
    return s


def cap(name):
    """CrossRef returns some surnames in block capitals; normalise them."""
    if name.isupper() and len(name) > 1:
        return name.capitalize()
    return name


def yr(m):
    for k in ('published-print', 'published-online', 'issued', 'created'):
        if k in m and m[k].get('date-parts', [[None]])[0][0]:
            return str(m[k]['date-parts'][0][0])
    return '????'


def authors(m):
    out = []
    for a in m.get('author', []) or []:
        fam = cap(clean(a.get('family', '')))
        giv = clean(a.get('given', ''))
        ini = ''.join(p[0].upper() + '.' for p in re.split(r'[\s-]+', giv) if p)
        out.append(f'{fam}, {ini}' if fam and ini else fam or clean(a.get('name', '')))
    return out


def short(names):
    fams = [n.split(',')[0] for n in names]
    if len(fams) == 1:
        return fams[0]
    if len(fams) == 2:
        return f'{fams[0]} and {fams[1]}'
    return f'{fams[0]} et al.'


rows, jcount, acount = [], Counter(), Counter()
for key, doi in ENTRIES:
    m = get(f'{API}/{urllib.parse.quote(doi)}')['message']
    au = authors(m)
    title = clean((m.get('title') or [''])[0])
    cont = clean((m.get('container-title') or [''])[0])
    vol, iss = clean(m.get('volume', '')), clean(m.get('issue', ''))
    pg = clean(m.get('page', '')).replace('-', '--')
    y = yr(m)
    jcount[cont.lower()] += 1
    for a in au:
        acount[a.split(',')[0].lower()] += 1
    bits = [', '.join(au[:-1]) + ' and ' + au[-1] if len(au) > 1 else (au[0] if au else '')]
    s = f"{bits[0]} ({y}) '{tex(title)}', {{\\it {tex(cont)}}}"
    if vol:
        s += f', Vol. {vol}'
    if iss:
        s += f', No. {iss}'
    if pg:
        s += f', pp. {pg}'
    s += f', doi: {doi}.'
    rows.append((key, short(au), y, s, cont))
    time.sleep(0.25)

for key, _, sh_, y, s in MANUAL:
    rows.append((key, sh_, y, s, 'McGraw-Hill'))

rows.sort(key=lambda r: (r[1].split()[0].lower(), r[2]))
with open('bibliography.tex', 'w', encoding='utf-8') as f:
    f.write('\\begin{thebibliography}{%d}\n\n' % len(rows))
    for key, sh_, y, s, _ in rows:
        f.write('\\bibitem[\\protect\\citeauthoryear{%s}{%s}]{%s}\n%s\n\n' % (sh_, y, key, s))
    f.write('\\end{thebibliography}\n')

print(f'{len(rows)} entries written to bibliography.tex\n')
print('journals with more than one entry:')
for k, v in jcount.most_common():
    if v > 1:
        flag = '  <-- OVER LIMIT' if v > 3 else ''
        print(f'  {v}  {k}{flag}')
print('\nauthors with more than one entry:')
for k, v in acount.most_common():
    if v > 1:
        flag = '  <-- OVER LIMIT' if v > 3 else ''
        print(f'  {v}  {k}{flag}')
recent = sum(1 for r in rows if r[2] >= '2025')
print(f'\n{recent} of {len(rows)} entries are from 2025 or later')
print('\nkeys:', ', '.join(r[0] for r in rows))
