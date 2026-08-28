"""Rebuild ai67.tex from the preserved preamble plus the revised sections."""
import io, os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', 'single column', 'ai67.tex'))
BAK = os.path.join(HERE, 'ai67_submitted.tex.bak')

src = io.open(PAPER, encoding='utf-8').read()
if not os.path.exists(BAK):
    shutil.copy(PAPER, BAK)
    print('backed up the submitted manuscript to', os.path.basename(BAK))

preamble = src[:src.index('\\begin{abstract}')]

ABSTRACT = r"""\begin{abstract}
Predictive modelling in finance has advanced faster than the machinery that converts a
prediction into a position. This paper specifies and tests a decision-theoretic
framework for that conversion. Latent market states are identified by a Gaussian
mixture model fitted on an expanding window; technical and market-state composites are
fused with state-dependent weights and scaled to a return metric; the fused signal
enters a constrained quadratic program with covariance shrinkage and an explicit
turnover penalty; and the resulting portfolio is scaled by a regime-dependent risk
budget. Evaluated on ten liquid multi-asset instruments over eighteen years under a
walk-forward protocol with transaction costs, the framework attains a Sharpe ratio of
0.88 against 0.63 for an equal-weighted benchmark and reduces maximum drawdown from
$-36.1$ to $-19.0$ per cent, at the cost of a lower compound return. It outperforms in
all four stress episodes in the sample and in 18 of 18 specifications examined, though
the Sharpe improvement is not statistically significant. The final 665 sessions
postdate every specification decision and are reported as a holdout; on them the
framework records a higher Sharpe ratio and a smaller maximum drawdown than all three
benchmarks considered. Ablation at matched average exposure isolates the source: timing
risk by market state improves both the Sharpe ratio and the drawdown with no change in
mean return, whereas the signal fusion layer contributes nothing and the
state-dependent weighting hypothesis is unsupported. The paper also identifies two
failure modes in frameworks of this kind: an absolute volatility target that never
binds, and a signal-to-allocation conversion in which mismatched units rather than
information determine the solution.
\end{abstract}

\KEYWORD{Portfolio Optimisation; Regime Detection; Risk Budgeting;
Decision-Theoretic Framework; Signal Fusion; Ablation Analysis; Asset Allocation}

\REF{}

\begin{bio}

\end{bio}


\maketitle

"""

parts = [preamble, ABSTRACT]
for f in ['new_sec1.tex', 'new_sec2.tex', 'new_sec3.tex', 'new_sec4.tex',
          'new_sec5.tex', 'new_sec6.tex']:
    parts.append(io.open(os.path.join(HERE, f), encoding='utf-8').read().rstrip() + '\n\n')

parts.append(r"""
\section*{Acknowledgements}
\label{sec:declarations}
The authors would like to thank the Fr.~C.~Rodrigues Institute of Technology, Vashi and
Fr.~C.~Rodrigues College of Engineering, Bandra for their support.

\section*{Author contributions}
\textbf{Anandkumar Pardeshi:} Conceptualization, Methodology, Software, Data curation,
Formal analysis, Writing -- original draft.\\
\textbf{Sujata Deshmukh:} Supervision, Validation, Writing -- review \& editing,
Research coordination.

\section*{Conflict of interest}
The authors declare no conflict of interest.

\section*{Data and code availability}
All market data used in this study are publicly available daily price and volume series
obtained from Yahoo Finance for the instruments listed in Section~3.1. The analysis code
implementing the framework, the specification sweep and the ablations is available from
the corresponding author on reasonable request.

\section*{AI tool usage declaration}
During the preparation of this work, the authors used [AUTHORS: NAME THE TOOL(S)
ACTUALLY USED] in order to improve the readability and language of the manuscript. After
using this tool, the authors reviewed and edited the content as needed and take full
responsibility for the content of the publication. No part of the analysis, the
generation of results, or the interpretation of findings was delegated to such tools.

""")

parts.append('\\label{sec:refs}\n')
parts.append(io.open(os.path.join(HERE, 'bibliography.tex'), encoding='utf-8').read())
parts.append('\\end{document}\n')

out = ''.join(parts)
# images live beside the .cls, so the document compiles from its own directory
out = out.replace('{single column/images/', '{images/')

io.open(PAPER, 'w', encoding='utf-8').write(out)
print('wrote %s  (%d lines, %d chars)' % (PAPER, out.count('\n') + 1, len(out)))

# quick structural check
keys = set(re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', out))
cited = set()
for m in re.finditer(r'\\cite[tp]?\{([^}]+)\}', out):
    cited |= {c.strip() for c in m.group(1).split(',')}
print('bibitems:', len(keys), ' cited:', len(cited))
print('cited but missing :', sorted(cited - keys) or 'none')
print('defined but unused:', sorted(keys - cited) or 'none')
labels = re.findall(r'\\label\{([^}]+)\}', out)
dup = {l for l in labels if labels.count(l) > 1}
print('duplicate labels  :', sorted(dup) or 'none')
refs = set(re.findall(r'\\ref\{([^}]+)\}', out))
print('dangling refs     :', sorted(refs - set(labels)) or 'none')
