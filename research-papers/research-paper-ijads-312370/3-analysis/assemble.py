"""Rebuild ai67.tex from the preserved preamble plus the revised sections."""
import io, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '2-revised-manuscript', 'ai67.tex'))
# the manuscript as submitted is archived under 1-original-submission/ai92.tex

src = io.open(PAPER, encoding='utf-8').read()

preamble = src[:src.index('\\begin{abstract}')]

ABSTRACT = r"""\begin{abstract}
This paper deals with a critical but relatively under-addressed question in the field of
quantitative finance: how to move from the activity of generating predictive financial
signals to the activity of executing decisions in asset allocation. The current
literature dwells mainly on predictive models based on technical indicators and
sentiment-based measures, and does not consider the systematic construction of
methodologies that convert those models into effective portfolio weights in dynamic
market regimes. A tripartite architecture is created through the proposed methodology.
First, a set of multimodal signals, including those derived from technical, sentiment and
volatility information, is synthesised. Thereafter, the latent market regimes are
outlined using Gaussian mixture models, which in effect define different probabilistic
conditions of the financial environment. The central mechanism is a regime-conditional
fusion rule that calibrates the impact of each signal by its assumed effectiveness within
the identified state, a hypothesis this study tests rather than presumes. This culminates
in a decision-theoretic core of optimisation, which incorporates shrinkage covariance
structures and regime-dependent risk-budgeting parameters in converting the synthesised
signal into a portfolio construction to be implemented. Evaluated on ten liquid
multi-asset instruments over eighteen years under a walk-forward protocol with
transaction costs, the strategy attains a Sharpe ratio of 0.88 against 0.63 for an
equal-weighted benchmark and lowers maximum drawdown from $-36.1$ to $-19.0$ per cent, at
the cost of a lower compound return. The advantage holds in all eighteen specifications
examined, in all four stress episodes, and in a held-out final block of 665 sessions that
postdates every specification decision, but it is not statistically significant. Ablation
at matched exposure locates the source: timing risk by market state contributes, whereas
the signal fusion layer does not. Considering this, this study is relevant to applied
decision sciences because it provides a clear and systematic way of closing the gap
between predictive signals and actionable investment decisions, together with evidence on
which parts of such a framework carry the result.
\end{abstract}

\KEYWORD{Portfolio Optimisation; Regime Detection; Sentiment Analysis;
Risk Budgeting; Decision-Theoretic Framework; Signal Fusion; Ablation Analysis;
Asset Allocation}

\REF{to this paper should be made as follows: Pardeshi, A. and Deshmukh, S.
(20XX) `From Signal Fusion to Asset Allocation: A Decision-Theoretic Model for
Portfolio Construction Under Regime-Based Sentiment and Volatility',
{\it Int. J. Applied Decision Sciences}, Vol. X, No. X, pp.xxx\textendash xxx.}

% Biographical notes: factual only -- affiliation and the research area of this
% paper. Expand with degrees, positions and publication record before upload.
\begin{bio}
Anandkumar Pardeshi is with the Department of Information Technology,
Fr.~C.~Rodrigues Institute of Technology, Vashi, Navi Mumbai, India. His research
interests include computational finance, machine learning for financial time
series, and decision-theoretic portfolio construction. He conceived and
implemented the framework reported in this paper.\vs{9}

\noindent Sujata Deshmukh is with the Department of Computer Engineering,
Fr.~C.~Rodrigues College of Engineering, Bandra, Mumbai, India. Her research
interests include machine learning, data analytics and their application to
financial decision problems. She supervised and validated the study reported in
this paper.
\end{bio}


\maketitle

"""

parts = [preamble, ABSTRACT]
for f in ['new_sec1.tex', 'new_sec2.tex', 'new_sec3.tex', 'new_sec4.tex',
          'new_sec5.tex']:
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
During the preparation of this work, the authors used Claude (Anthropic) and ChatGPT
(OpenAI) in order to improve the readability and language of the manuscript. After
using these tools, the authors reviewed and edited the content as needed and take full
responsibility for the content of the publication. No part of the analysis, the
generation of results, or the interpretation of findings was delegated to such tools.

""")

parts.append('\\label{sec:refs}\n')
parts.append(io.open(os.path.join(HERE, 'bibliography.tex'), encoding='utf-8').read())
parts.append('\\end{document}\n')

out = ''.join(parts)

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
