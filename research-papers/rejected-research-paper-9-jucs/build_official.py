# Splice the verified manuscript body into the official J.UCS V5 template.
import re, io

src = open('jucs_decision_value.tex', encoding='utf-8').read()

# --- extract front-matter pieces from the self-contained version -------------
abstract = re.search(r'\\textbf\{Abstract:\}(.*?)\\smallskip\s*\\noindent\\textbf\{Key Words:\}',
                     src, re.S).group(1).strip()
keywords = re.search(r'\\textbf\{Key Words:\}(.*?)\\smallskip\s*\\noindent\\textbf\{Category:\}',
                     src, re.S).group(1).strip()
category = re.search(r'\\textbf\{Category:\}(.*?)\\par\}', src, re.S).group(1).strip()

# --- body (Introduction .. just before Acknowledgements) and the back-matter --
i_intro = src.index(r'\section{Introduction}')
i_ack   = src.index(r'\section*{Acknowledgements}')
i_end   = src.index(r'\end{document}')
body       = src[i_intro:i_ack]
backmatter = src[i_ack:i_end]          # acks + AI decl + data + conflict + bib

# --- make figures fit whatever \textwidth the official geometry yields --------
body = body.replace('width=11.5cm', r'width=\linewidth')
body = body.replace('width=12cm',   r'width=\linewidth')
# wrap the (first, = pipeline) TikZ picture so it scales to the column
body = body.replace(r'\begin{tikzpicture}[',
                    '\\resizebox{\\linewidth}{!}{%\n\\begin{tikzpicture}[', 1)
body = body.replace(r'\end{tikzpicture}', r'\end{tikzpicture}}', 1)

PREAMBLE = r"""% =============================================================================
% Official J.UCS V5 template build of:
% "Accounting for Decision Value: Calibration and Selection over Prediction
%  in a Deployed Equity-Forecasting System."
% Compile with LuaLaTeX (template uses fontspec + Times New Roman):
%     lualatex jucs_decision_value_official.tex   (x2 for refs)
% Requires jucs2e.sty in this folder (copied from the official template).
% =============================================================================
\documentclass[10pt, a4paper, oneside]{article}
\usepackage[hidelinks]{hyperref}
\usepackage{jucs2e}
\usepackage{graphicx}
\usepackage{url}
\usepackage{ulem}
\usepackage{mathtools}
\usepackage{amssymb}
\let\proof\relax\let\endproof\relax
\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{enumitem}
\usepackage{scalerel}
\usepackage{setspace}
\usepackage[strict]{changepage}
\usepackage{caption}
\usepackage[letterspace=-50]{microtype}
\usepackage{fontspec}
\usepackage{afterpage}
\usepackage{ragged2e}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,fit,backgrounds}

\setmainfont{Times New Roman}

\usepackage{titlesec}
\titleformat*{\section}{\Large\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}
\titleformat*{\subsubsection}{\normalsize\bfseries\itshape}

\renewcommand{\baselinestretch}{0.9}
\graphicspath{{./figures/}}
\usepackage[textwidth=8cm, margin=0cm, left=4.6cm, right=4.2cm, top=3.9cm, bottom=6.8cm, a4paper, headheight=0.5cm, headsep=0.5cm]{geometry}
\usepackage{fancyhdr}
\usepackage[format=plain, labelfont=it, textfont=it, justification=centering]{caption}
\usepackage{breakcites}
\usepackage{microtype}

\apptocmd{\frame}{}{\justifying}{}
\urlstyle{same}
\pagestyle{fancy}

% --- additions used by the manuscript body -----------------------------------
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\theoremstyle{plain}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newcommand{\ece}{\mathrm{ECE}}
\newcommand{\Pup}{\hat p}

% --- journal metadata (placeholders; editor assigns final values) ------------
\newcommand\jucs{{Journal of Universal Computer Science}}
\newcommand\jucsvol{vol. XX, no. X (2026)}
\newcommand\jucspages{XXXX-XXXX}
\newcommand\jucssubmitted{XX/XX/2026}
\newcommand\jucsaccepted{XX/XX/2026}
\newcommand\jucsappeared{XX/XX/2026}
\newcommand\jucslicence{ CC BY 4.0}
\newcommand\startingPage{1}
\setcounter{page}{\startingPage}

\newcommand\paperauthor{{Pardeshi, A., Deshmukh, S.: }}
\newcommand\papertitle{Accounting for Decision Value: Calibration and Selection over Prediction in a Deployed Equity-Forecasting System}
\header{\paperauthor Accounting for Decision Value}

% jucs2e hardcodes a sample running head; override its inner-page centre header
% with the real one (vendor jucs2e.sty left unmodified).
\fancyhead[OC]{%
  {\ifnum\thepage=\startingPage
   \begin{singlespace}\fontsize{8pt}{5pt}\selectfont{\itshape{\jucs, \jucsvol, \jucspages \\ submitted: \jucssubmitted, accepted: \jucsaccepted, appeared: \jucsappeared \jucslicence}}\vspace{-5mm}\end{singlespace}
   \else
   \lsstyle{\fontsize{8pt}{5pt}\selectfont{\itshape{Pardeshi, A., Deshmukh, S.: Accounting for Decision Value}}}\vspace{-0.5mm}
   \fi}%
}
"""

TITLEBLOCK = r"""\title{{\fontsize{14pt}{16pt}\selectfont{\vspace*{-3mm}\papertitle\vspace*{-1mm}}}}

\author{{\bfseries\fontsize{10pt}{12pt}\selectfont{Anandkumar Pardeshi}} \\
   {\fontsize{9pt}{11pt}\selectfont{(Department of Computer Science and Engineering, Fr.\ C.\ Rodrigues Institute of Technology,\\
   University of Mumbai, Navi Mumbai, India\\
   \orcid{0000-0002-1825-0097},
   anand.pardeshi@fcrit.ac.in)}}
   \and
   {\bfseries\fontsize{10pt}{12pt}\selectfont{Sujata Deshmukh}}\\
   {\fontsize{9pt}{11pt}\selectfont{(Department of Computer Engineering, Fr.\ C.\ Rodrigues Institute of Technology,\\
   University of Mumbai, Navi Mumbai, India\\
   \orcid{0000-0001-5109-3700},
   sujata.deshmukh@fcrit.ac.in)}}
}

\label{first}
\maketitle"""

def env(name, content, lead=""):
    return ("{\\fontfamily{ptm}\\selectfont\n\\begin{%s}\n"
            "{\\fontsize{9pt}{11pt}\\selectfont{%s\n"
            "%s}}\n\\end{%s}}\n") % (name, lead, content, name)

doi_env = ("{\\fontfamily{ptm}\\selectfont\n\\begin{doi}\n"
           "{\\fontsize{9pt}{11pt}\\selectfont{\n"
           "10.3897/jucs.\\textless SubmissionNumber\\textgreater}}\n\\end{doi}}\n")

out = (PREAMBLE
       + "\n\\begin{document}\n\n"
       + TITLEBLOCK + "\n\n"
       + env("abstract", abstract, lead="\\vspace*{-2mm}")
       + env("keywords", keywords)
       + env("category", category)
       + doi_env
       + "\n"
       + body
       + backmatter
       + "\\end{document}\n")

open('jucs_decision_value_official.tex', 'w', encoding='utf-8').write(out)
print("wrote jucs_decision_value_official.tex  (%d chars)" % len(out))
print("abstract %d chars | keywords: %s | category: %s" % (len(abstract), keywords[:40], category))
