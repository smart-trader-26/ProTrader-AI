%%==================    ecit.cls  ==============================
%%  LaTex2e class file for ECTI Transactions
%%  Modify from conference.cls(ITC-CSCC 2002) and IEEEtran.sty
%%  by Phakphoom Boonyanant (!PB) phakphoom@nectec.or.th
%%  all changes marked with !PB
%%
%%  CHANGES: 16.10.2002
%%  1. minor changes in \maketitle, \abstract, \keywords, \section and \caption
%%       according to the ECTI format:
%%          - Use \name instead of \author
%%          - Use \address for authors' affiliations
%%          - Use \uppercase font for \section
%%          - Use italic font for \caption
%%  2. Add a slight modification of 'citesort.sty' so that many references
%%       will be sorted and included in one bracket, for example, [1,3,4] or [1-5].
%%
%%
%%  USAGE:
%%            \documentclass[10pt,twocolumn]{ecti}
%%                      \begin{document}
%%            \title{..}
%%            \name{..}
%%                      \address{..}
%%            \maketitle
%%            \begin{abstract}...\end{abstract}
%%            \begin{keywords}...\end{keywords}
%%                      \section{..}
%%                      \cite{name1,name2,...}
%%%===========================================================
%%% For ITC-CSCC 2002
%%%
%%% (hopefully) minor changes for \LaTeXe
%%% by Peter N"uchter (!PN) peter@mwt.e-technik.uni-ulm.de
%%% all changes marked with !PN
%%%
%%% WARNING: math font problems and workaround:
%%%          Standard math commands like
%%%      $\mathbf{N}(0,P(0))$
%%%      will not work properly. The reason is, that this is
%%%      a simple hack of an old LaTeX2.09 style files.
%%%      To get it running, I used the packages
%%%      'rawfonts' and 'oldlfont'
%%%      They are responsible for the font problems.
%%%      This is a known feature, see
%%%      http://www.tex.ac.uk/CTAN/latex/bugs.html
%%%
%%%  WORKAROUND:
%%%  Use an additional pair of braces, like
%%%  {\mathbf{y}}
%%%  to introduce an additional level of scope.
%%%  Examples:
%%%      $\mathbf{N}(0,P(0))$ but ${\mathbf{N}}(0,P(0))$
%%%      $\mathcal{N}(0,P(0))$ but ${\mathcal{N}}(0,P(0))$
%%%      $\mathcal{G}_{t}$ but ${\mathcal{G}}_{t}$
%%%      $\mathfrak{G}_t$ but ${\mathfrak{G}}_t$
%%%
%%%
%%% Changes: 10.01.1996
%%%  \@maxsep removed
%%%  \@dblmaxsep removed
%%%  \typeout{Upgraded to `IEEEtran.cls' <10-Jan.-1996>.} added
%%%  \usepackage{rawfonts} added
%%%  \usepackage{oldlfont} added
%%%  \renewcommand{\normalsize}{\fontsize{10}{12}\selectfont} added
%%%  \AtBeginDocument{\parindent1.0em} added
%%%
%%% Changes: 14.05.1996
%%%  \setlength - bock added.
%%%  \renewcommand{\normalsize}{... hopefully fixed!
%%%  \topsep defs inserted at pointsize declarations
%%%     ^^-- this is a little bit tricky. I fiddled around until
%%%      I nearly obtained the same result as with LaTeX209.
%%%      I think we can live with that now?
%%%
%%% Changes: 1.5.1997
%%%  Inclusion of math-font-problem-section on top of this file!
%%%
%%%
%%% Resulting Differenzes between \LaTeX209 and \LaTeX:
%%%  Slightly different formatting.
%%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  IEEEtran.sty  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%=========================================================================
%%%%  @LaTeX-style-file{
%%%%    author          = "Gerry Murray and Silvano Balemi",
%%%%    version         = "2.5.4",
%%%%    date            = "30-August-1993",
%%%%    filename        = "IEEEtran.sty",
%%%%    address         = "Automatic Control Lab, ETH Zurich, Switzerland"
%%%%    email           = "balemi@aut.ee.ethz.ch,
%%%%    keywords        = "LaTeX, IEEE Transactions, publisher style",
%%%%    mainserver      = "via E-mail from balemi@aut.ee.ethz.ch ",
%%%%    docstring       = "This is a LaTeX style file for preparation
%%%%                       of submission and final versions of articles
%%%%                       to be published in IEEE Transactions.
%%%%                       The default font size is 10 points but 9pt,
%%%%                       11pt and 12pt are available too.
%%%%
%%%%                        Usage:
%%%%                         \documentstyle[..,Xpt,twoside]{IEEEtran}
%%%%                         \author{..}
%%%%                         \title{..}
%%%%                         \maketitle
%%%%                         \begin{abstract}...\end{abstract}
%%%%                         \begin{keywords}...\end{keywords}
%%%%                         ...
%%%                         \begin{biography}{Author's name}...\end{biography}
%%%%                         \end{document}
%%%%
%%%%                        The command "\markboth{leftTEXT}{rightTEXT}"
%%%%                       puts the text "leftTEXT" and "rightTEXT"
%%%%                       with the page number on the even resp. odd
%%%%                       headers when the "twoside" option is selected.
%%%%                       (Attention: header of title page is
%%%%                       always "leftTEXT" as it bears the journal name).
%%%%                       In order to remove headers, insert
%%%%                          \thispagestyle{empty}\pagestyle{empty} or
%%%%                          \thispagestyle{plain}\pagestyle{plain}
%%%%                       right after the \maketitle command!!
%%%%
%%%%                       See also IEEE.bst and IEEEsample.tex",
%%%%  }
%%%% =========================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\typeout{Upgraded to `ecti.cls' <16-Oct.-2002>} %!PB
\typeout{Rev. <16-Oct.-2002>} %!PB

\usepackage{rawfonts} %!PN
\usepackage{oldlfont} %!PN
%!PN, 14.05.96 \renewcommand{\normalsize}{\fontsize{10}{12}\selectfont} %!PN
% this always resulted in a 10pt doc ....., sorry
\renewcommand{\normalsize}{\@normalsize} %!PN

\AtBeginDocument{\parindent4.22mm} %!PN


%% define new needed boolean variables
\newif\if@tmptwocolumn  \@tmptwocolumnfalse
\newif\if@draftversion  \@draftversionfalse
\newif\if@technote      \@technotefalse

\def\@ptsize{10}
\@namedef{ds@9pt}{\def\@ptsize{9}}
\@namedef{ds@10pt}{\def\@ptsize{10}}
\@namedef{ds@11pt}{\def\@ptsize{11}}
\@namedef{ds@12pt}{\def\@ptsize{12}}
\@namedef{ds@twoside}{\@twosidetrue \@mparswitchtrue}
\@namedef{ds@draft}{\global\@draftversiontrue} % sets variable for draft
\@namedef{ds@technote}{\global\@technotetrue}  % sets variable for techn. note
%% the file twocolumn.sty is not read as it changes \textwidth.
%% Next line sets @tmptwocolumn to true which lets later execute \twocolumn
\@namedef{ds@twocolumn}{\global\@tmptwocolumntrue }
\@options

%%%%% \input IEEE\@ptsize.sty\relax  %%%% FONT DEFINITION BELOW (NO FILE READ)

%% change sizes and margins !PB
%% A4 size = wxh = 8.27in x 11.69in
%% Margin: Top 1.37in, Bottom 1.65in, Left 0.86in, Right 0.86in
%% Top = 1+\voffset+\topmargin+\headsep+\headheight = 1-0.37+0+0.2+0.17 = 1in
%% Bottom = h-Top-\textheight = 11.69-1-9.69 = 1.65in
%% Left = 1+\hoffset+\oddsidemargin = 1+0-0.2 = 0.8in
%% Right = w-Left-\textwidth = 8.27-0.8-6.67 = 0.8in
\topmargin              0in
\voffset                -0.5in
\oddsidemargin  -0.2in
\evensidemargin -0.2in
\textheight      9.65in
\textwidth       6.57in
\columnsep        5.0mm
\parindent        0.1em
\headsep          0.45in
\headheight       0.17in
\lineskip           1pt
\normallineskip     1pt
\def\baselinestretch{1}

\if@draftversion
\topmargin       -6.0mm
\oddsidemargin      0mm
\evensidemargin     0mm
\textheight     223.5mm
\textwidth      161.0mm
\fi

\marginparsep      40pt
\marginparwidth    20pt
\marginparpush     25pt

\partopsep          \z@
\topsep           0.4ex
\parsep             \z@
\itemsep            \z@
%% see further below for lists


%% FONT DEFINITION: avoids having to read in font files.
%%
%% Check if we have selected 9 points
\def\@tempa{9}\ifx\@ptsize\@tempa
\typeout{-- This is a 9 point document}
\def\@normalsize{\@setsize\normalsize{10.7pt}\ixpt\@ixpt
\abovedisplayskip 1em plus2pt minus5pt\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ plus3pt\belowdisplayshortskip .6em plus3pt minus3pt
\topsep \belowdisplayshortskip%!PN
}
\def\small{\@setsize\small{9.12pt}\viiipt\@viipt}
\def\footnotesize{\@setsize\footnotesize{8.15pt}\viipt\@vipt}
\def\scriptsize{\@setsize\scriptsize{8pt}\vipt\@vpt}
\def\tiny{\@setsize\tiny{5pt}\vpt\@vpt}
\def\large{\@setsize\large{12pt}\xpt\@xpt}
\def\Large{\@setsize\Large{14pt}\xiipt\@xiipt}
\def\LARGE{\@setsize\LARGE{18pt}\xivpt\@xivpt}
\def\huge{\@setsize\huge{22pt}\xviipt\@xviipt}
\def\Huge{\@setsize\Huge{25pt}\xxpt\@xxpt}
\fi
%%
%% Check if we have selected 10 points
\def\@tempa{10}\ifx\@ptsize\@tempa
\typeout{-- This is a 10 point document}
\def\@normalsize{\@setsize\normalsize{11.9pt}\xpt\@xpt
\abovedisplayskip 1em plus2pt minus5pt\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ plus3pt\belowdisplayshortskip .6em plus3pt minus3pt
\topsep \belowdisplayshortskip%!PN
}
\def\small{\@setsize\small{9.2pt}\viiipt\@viiipt}
\def\footnotesize{\@setsize\footnotesize{8.8pt}\viiipt\@viiipt}
\def\scriptsize{\@setsize\scriptsize{8pt}\viipt\@viipt}
\def\tiny{\@setsize\tiny{6pt}\vpt\@vpt}
\def\large{\@setsize\large{11.9pt}\xpt\@xpt}
\def\Large{\@setsize\Large{12pt}\xiipt\@xiipt}
\def\LARGE{\@setsize\LARGE{14pt}\xivpt\@xivpt}
\def\huge{\@setsize\huge{20pt}\xxpt\@xxpt}
\def\Huge{\@setsize\Huge{28pt}\xxvpt\@xxvpt}
\fi
%%
%% Check if we have selected 11 points
\def\@tempa{11}\ifx\@ptsize\@tempa
\typeout{-- This is an 11 point document}
\def\@normalsize{\@setsize\normalsize{13.6pt}\xipt\@xipt
\abovedisplayskip 1em plus2pt minus5pt\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ plus3pt\belowdisplayshortskip .6em plus3pt minus3pt
\topsep \belowdisplayshortskip%!PN
}
\def\small{\@setsize\small{12pt}\xpt\@xpt}
\def\footnotesize{\@setsize\footnotesize{11pt}\ixpt\@ixpt}
\def\scriptsize{\@setsize\scriptsize{9.5pt}\viiipt\@viiipt}
\def\tiny{\@setsize\tiny{7pt}\vipt\@vipt}
\def\large{\@setsize\large{14pt}\xiipt\@xiipt}
\def\Large{\@setsize\Large{18pt}\xivpt\@xivpt}
\def\LARGE{\@setsize\LARGE{22pt}\xviipt\@xviipt}
\def\huge{\@setsize\huge{25pt}\xxpt\@xxpt}
\def\Huge{\@setsize\Huge{30pt}\xxvpt\@xxvpt}
\fi
%%
%% Check if we have selected 12 points
\def\@tempa{12}\ifx\@ptsize\@tempa
\typeout{-- This is a 12 point document}
\def\@normalsize{\@setsize\normalsize{14pt}\xiipt\@xiipt
\abovedisplayskip 1em plus3pt minus6pt\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ plus3pt\belowdisplayshortskip .6em plus4pt minus4pt
\topsep \belowdisplayshortskip%!PN
}
\def\small{\@setsize\small{11.4pt}\xpt\@xpt}
\def\footnotesize{\@setsize\footnotesize{10pt}\ixpt\@ixpt}
\def\scriptsize{\@setsize\scriptsize{9pt}\viiipt\@viiipt}
\def\tiny{\@setsize\tiny{8pt}\vipt\@vipt}
\def\large{\@setsize\large{18pt}\xivpt\@xivpt}
\def\Large{\@setsize\Large{22pt}\xviipt\@xviipt}
\def\LARGE{\@setsize\LARGE{25pt}\xxpt\@xxpt}
\def\huge{\@setsize\huge{30pt}\xxvpt\@xxvpt}
\let\Huge=\huge
\fi


%% Definitions for PARAGRAPHS
%%
\abovedisplayskip       0pt plus2pt minus5pt%
\belowdisplayskip       \abovedisplayskip
\abovedisplayshortskip  0pt plus3pt%
\belowdisplayshortskip  0pt plus3pt minus3pt%

%% Definitions for LISTS
%%

%% Change aspect of lists with
%% 1) \itemindent,  label indentation  wrt to left list margin
%% 2) \leftmargini, the indentation of the whole list (on left, first level)
\itemindent        -1em
\leftmargini        2em
%\itemindent         2em  % Alternative values: sometimes used..
%\leftmargini        0em
\leftmarginii       1em
\leftmarginiii    1.5em
\leftmarginiv     1.5em
\leftmarginv      1.0em
\leftmarginvi     1.0em
\labelsep           5pt
\leftmargin\leftmargini
\labelwidth         \z@

\def\@listI{\leftmargin\leftmargini} \@listI
\def\@listi{\leftmargin\leftmargini  \topsep \z@ plus 1pt minus 1pt}
\def\@listii{\leftmargin\leftmarginii\labelwidth\leftmarginii
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listiii{\leftmargin\leftmarginiii\labelwidth\leftmarginiii
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listiv{\leftmargin\leftmarginiv\labelwidth\leftmarginiv
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listv{\leftmargin\leftmarginv\labelwidth\leftmarginv
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listvi{\leftmargin\leftmarginvi\labelwidth\leftmarginvi
    \advance\labelwidth-\labelsep    \topsep \z@}

\def\labelenumi{\theenumi.}     \def\theenumi{\arabic{enumi}}
\def\labelenumii{\theenumii}  \def\theenumii{\alph{enumii}}
\def\labelenumiii{\theenumiii.} \def\theenumiii{\roman{enumiii}}
\def\labelenumiv{\theenumiv.}   \def\theenumiv{\Alph{enumiv}}
\def\p@enumii{\theenumi}
\def\p@enumiii{\theenumi\theenumii}
\def\p@enumiv{\p@enumiii\theenumiii}

\def\labelitemi{$\scriptstyle\bullet$}
\def\labelitemii{\bf --}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}

%% \itemindent is set to \z@ by list, so define new temporary variable
\newdimen\tmpitemindent
\def\verse{\let\\=\@centercr
    \list{}{\itemsep\z@ \itemindent -1.5em \listparindent \itemindent
    \rightmargin\leftmargin\advance\leftmargin 1.5em}\item[]}
\let\endverse\endlist
\def\quotation{\list{}{\listparindent 1.5em \itemindent\listparindent
    \rightmargin\leftmargin \parsep 0pt plus 1pt}\item[]}
\let\endquotation=\endlist
\def\quote{\list{}{\rightmargin\leftmargin}\item[]}
\let\endquote=\endlist
\def\@mklab#1{#1}
\def\description{\tmpitemindent\itemindent\list{}{\itemindent\tmpitemindent
    \labelwidth\z@\def\makelabel##1{\hspace\labelsep\it{##1}}}}
\def\enddescription{\endlist\par}
\def\enumerate{\tmpitemindent\itemindent\ifnum \@enumdepth >3 \@toodeep\else
    \advance\@enumdepth \@ne \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
    \list{\csname label\@enumctr\endcsname}{\itemindent\tmpitemindent
    \usecounter{\@enumctr}\def\makelabel##1{\hspace\labelsep\hfil{##1}}}\fi}
\def\endenumerate{\endlist\par}
\def\itemize{\tmpitemindent\itemindent\ifnum \@itemdepth >3 \@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \list{\csname\@itemitem\endcsname}{\itemindent\tmpitemindent
    \def\makelabel##1{\hspace\labelsep\hfil\it{##1}}}\fi}
\def\enditemize{\endlist\par}

\newif\if@restonecol
\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
    \else \newpage \fi \c@page\z@}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi}

\arraycolsep     5pt
\tabcolsep       6pt
\arrayrulewidth .4pt
\doublerulesep   2pt
\tabbingsep\labelsep

%% FOOTNOTES
%
\skip\footins 10pt plus 4pt minus 2pt
\footnotesep 7pt
\footskip 10mm
\skip\@mpfootins = \skip\footins
\fboxsep = 3pt \fboxrule = .4pt
\long\def\@makefntext#1{#1}
\def\footnoterule{\noindent \rule[0.5em]{5cm}{0.15mm}}

\if@technote
   \setcounter{secnumdepth}{3}
\else
   \setcounter{secnumdepth}{4}
\fi

\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]

\def\thesection{\arabic{section}.\hskip -0.5em}
\def\thesubsection{\thesection \hskip 0.6em \arabic{subsection} \hskip -0.8em }
\def\thesubsubsection{\thesubsection...\arabic{subsubsection}}
\def\thesubsubsectiondis{\thesubsubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\alph{paragraph}}
\def\theparagraphdis{\thesubsubsectiondis.\alph{paragraph}}

\def\theequation{\arabic{equation}}

%% LIST OF FIGURES AND TABLES
%
\def\@pnumwidth{1.55em}
\def\@tocrmarg {2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{3}

\def\tableofcontents{\section*{Contents}\@starttoc{toc}}
\def\l@section#1#2{\addpenalty{\@secpenalty} \addvspace{1.0em plus 1pt}
    \@tempdima 1.8em \begingroup \parindent \z@ \rightskip \@pnumwidth
    \parfillskip-\@pnumwidth \bf\leavevmode #1\hfil\hbox to\@pnumwidth{\hss #2}
    \par \endgroup}
\def\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\def\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\def\listoffigures{\section*{List of Figures}\@starttoc{lof}}
\def\l@figure{\@dottedtocline{1}{1em}{1em}}
\def\listoftables{\section*{List of Tables}\@starttoc{lot}}
\let\l@table\l@figure

%% Definitions for floats
%%
%% Normal Floats                      %% Double Column Floats
\floatsep 12pt plus 2pt minus 2pt     \dblfloatsep 12pt plus 2pt minus 2pt
\textfloatsep 20pt plus 2pt minus 4pt \dbltextfloatsep 20pt plus 2pt minus 4pt
%\@maxsep 20pt                         \@dblmaxsep 20pt %!PN
\@fptop 0pt plus 1fil                 \@dblfptop 0pt plus 1fil
\@fpsep 8pt plus 2fil                 \@dblfpsep 8pt plus 2fil
\@fpbot 0pt plus 1fil                 \@dblfpbot 0pt plus 1fil
\def\topfraction{1.0}                 \def\dbltopfraction{1.0}
\def\floatpagefraction{0.8}           \def\dblfloatpagefraction{0.8}
\setcounter{topnumber}{2}             \setcounter{dbltopnumber}{4}
%
\intextsep 12pt plus 2pt minus 2pt
\setcounter{bottomnumber}{2}
\def\bottomfraction{.4}
\setcounter{totalnumber}{4}
\def\textfraction{.2}


%% redefine CAPTION%%%%%%%

\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname ext@#1\endcsname}{#1}{%
   \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}}
   \begingroup \@parboxrestore \normalsize
     \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}{#1}\par
   \endgroup}

\def\tablestring{table}
\def\figurestring{figure}

\newlength{\captionindent}
\long\def\@makecaption#1#2#3{
\ifx\figurestring#3 %%%remove \captionindent and use italic font !PB
    \vskip 5pt
    \setbox\@tempboxa\hbox{\normalsize \em #1~ #2}
    \ifdim \wd\@tempboxa >\hsize
       \setbox\@tempboxa\hbox{\normalsize \em #1~ }
       \parbox[t]{\hsize}{\normalsize \em \unhbox\@tempboxa#2}
       \else \hbox to\hsize{\normalsize \em \hfil\box\@tempboxa\hfil}
\fi
\else
    \ifx\tablestring#3 %%%remove \captionindent and use italic font !PB
       \vskip 5pt
       \setbox\@tempboxa\hbox{\normalsize \em #1~ #2}
       \ifdim \wd\@tempboxa >\hsize
       \setbox\@tempboxa\hbox{\normalsize \em #1~ }
       \parbox[t]{\hsize}{\normalsize \em \unhbox\@tempboxa#2}
       \else \hbox to\hsize{\normalsize \em \hfil\box\@tempboxa\hfil}
       \fi
   \else
        \begin{center}{\normalsize #1}\\{\normalsize \sc #2}\end{center}
        \vskip 8pt
   \fi
\fi}

\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\textit{\textbf{Fig.\thefigure:}}} %%%Use boldface and italic font!PB
\def\figure{\@float{figure}}
\let\endfigure\end@float
\@namedef{figure*}{\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}

\newcounter{table}
%\def\thetable{\@Roman\c@table}
\def\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\textit{\textbf{Table \thetable:}}} %%%Use boldface and italic font!PB
\def\table{\@float{table}}
\let\endtable\end@float
\@namedef{table*}{\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\@Rtop\relax\let\@Ltop\relax\let\@Rbot\relax\let\@Lbot\relax
\let\pot@jobname\relax

\def\runninghead#1{%
\gdef \@runninghead{#1}}

\def\ps@ECTI{%
\let\@mkboth\@gobbletwo
\def\@oddhead{\@Ltop
  {\scriptsize \@runninghead \hfil \thepage}%
  \@Rtop}%
\def\@evenhead{\@Ltop
  {\scriptsize \thepage \hfil ECTI TRANSACTIONS ON COMPUTER AND INFORMATION TECHNOLOGY\, VOL.\@vol, NO.\@no\ \month@year}%
  \@Rtop}%
\def\@evenfoot{\@Lbot\hfil\@Rbot}%
\let\@oddfoot\@evenfoot
}

\newif\if@outputMA\@outputMAtrue

\def\vol#1{%
\gdef\hold@vol{#1}%
\gdef\@vol{#1}%
}
\def\hold@vol{2004}
\gdef\@vol{11}

\def\no#1{\gdef\@no{#1}}
%CHANGE VOL:
\def\@no{1}


\AtBeginDocument{%
\@tempcnta \hold@vol\relax
%\advance \@tempcnta 1\relax
%CHANGE DATE
\xdef\month@year{May 2017}%
}
%% Original Code
%%\xdef\month@year{\ifcase\@no \or
 %% JANUARY\or FEBRUARY\or MARCH\or APRIL\or MAY\or JUNE\or
 %% JULY\or AUGUST\or SEPTEMBER\or OCTOBER\or NOVEMBER\or DECEMBER\fi
  %%\ \the\@tempcnta}%
%%}

\long\def\title{\@ifnextchar[%% ]
{\d@title}{\s@title}}
\def\d@title[#1]#2{%
\gdef\rh@title{\uppercase{#1}}\gdef\@title{#2\@titlenotemark}}
\def\s@title#1{\gdef\rh@title{\uppercase{#1}}\gdef\@title{#1\@titlenotemark}}
\def\@title{}\def\rh@title{}

\def\titlenote#1{\gdef\@titlenote{#1}}
\def\@titlenote{}

\newcount\tn@cnt
\def\title@notation{%
\ifx\@titlenote\empty\else
  \footnotetext[\the\tn@cnt]{\@titlenote}%
\fi
}

\def\@titlenotemark{%
\ifx\@titlenote\empty\else
  \unskip\nolinebreak\footnotemark[\the\tn@cnt]%
\fi
}
\def\@outputrunninghead{%
\if@letter
  \gdef\r@head{LETTER}%
\else
  \ifnum\total@mbr=\z@
   \gdef\r@head{\rh@title}%
  \else
   \ifnum\total@mbr=\@ne
    \setbox\@tempboxa\hbox{\spaceskip=100mm \csname
@Rauthorname1\endcsname}%
    \ifdim \wd\@tempboxa < 100mm %% >
     \xdef\r@head{\csname @Rauthorname1\endcsname\unskip:\ \rh@title}%
    \else
     \def\@rhead##1 ##2\@nd{\gdef\r@head{{##2}\unskip:\
\rh@title}}%
     \edef\@tempname{\csname @Rauthorname1\endcsname}%
     \expandafter\@rhead\@tempname\@nd\relax
    \fi
   \else
    \ifnum\total@mbr=\tw@
     \setbox\@tempboxa\hbox{\spaceskip=100mm
      \csname @Rauthorname1\endcsname\csname @Rauthorname2\endcsname}%
     \ifdim \wd\@tempboxa < 200mm %% >
      \xdef\r@head{\csname @Rauthorname1\endcsname\
        and \csname @Rauthorname2\endcsname\unskip:\ \rh@title}%
     \else
      \def\@rhead##1 ##2 ##3 ##4\@nd{\gdef\r@head{{##2}\
        and {##4}\unskip:\ \rh@title}}%
      \edef\@tempname{\csname @Rauthorname1\endcsname\space
                      \csname @Rauthorname2\endcsname}%
      \expandafter\@rhead\@tempname\@nd\relax
     \fi
    \else
     \setbox\@tempboxa\hbox{\spaceskip=100mm \csname
@Rauthorname1\endcsname}%
     \ifdim \wd\@tempboxa < 100mm %% >
      \xdef\r@head{\csname @Rauthorname1\endcsname\unskip\ et al.:\
\rh@title}%
     \else
      \def\@rhead##1 ##2\@nd{%
       \gdef\r@head{{##2}\unskip\ et al.:\ \rh@title}}%
      \edef\@tempname{\csname @Rauthorname1\endcsname}%
      \expandafter\@rhead\@tempname\@nd\relax
     \fi
    \fi
   \fi
  \fi
  \chk@rhead
\fi
}
\def\r@head{}

\def\chk@rhead{%
\setbox\@tempboxa\hbox{\scriptsize r@head{\csname @Rauthorname1\endcsname\unskip\ et al.:\
\rh@title}}%% <
\ifdim \wd\@tempboxa > 182mm %% \textwidth176mm
   \ECTI@warning{Title of header is too wide. make it shorter}%
\fi
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\if@twoside
\def\ps@headings{
    \def\@oddhead{\hbox{}\scriptsize \rightmark \hfil \thepage}
    \def\@evenhead{\scriptsize\thepage \hfil Y \leftmark\hbox{}}
    \if@draftversion
         \def\@oddfoot{\scriptsize\@date\hfil DRAFT}
         \def\@evenfoot{\scriptsize DRAFT\hfil\@date}
    \else
         \def\@oddfoot{}\def\@evenfoot{}
    \fi}
\else
\def\ps@headings{
    \def\@oddhead{\hbox{}\scriptsize XXXX \leftmark \hfil \thepage}
    \def\@evenhead{\scriptsize \thepage \hfil YYYY \leftmark\hbox{}}
    \if@draftversion
         \def\@oddfoot{\scriptsize \@date \hfil DRAFT}
    \else
         \def\@oddfoot{}
    \fi
    \def\@evenfoot{}}
\fi
\def\ps@titlepagestyle{
    \def\@oddhead{\hbox{}\scriptsize ECTI TRANSACTIONS ON COMPUTER AND INFORMATION TECHNOLOGY \leftmark \hfil \thepage}
    \def\@evenhead{\scriptsize\thepage \hfil \rightmark\hbox{}}
    \if@draftversion
         \def\@oddfoot{\scriptsize \@date\hfil DRAFT}
         \def\@evenfoot{\scriptsize DRAFT\hfil \@date}
    \else
         \def\@oddfoot{}\def\@evenfoot{}
    \fi}
\def\rightmark{}\def\leftmark{}


%% Defines the command for putting the header. footernote{TEXT} is the same
%% as markboth{TEXT}{TEXT}. Here for compatibility with other style files.
\def\markboth#1#2{\def\leftmark{\uppercase{#1}}\def\rightmark{\uppercase{#2}}}
\def\footernote#1{\markboth{#1}{#1}}

\def\today{\ifcase\month\or
    January\or February\or March\or April\or May\or June\or
    July\or August\or September\or October\or November\or December\fi
    \space\number\day, \number\year}

%% BIBLIOGRAPHY

%% from citesort.sty, a little customized !PB
\newcount\@minsofar
\newcount\@min
\newcount\@cite@temp
\def\@citex[#1]#2{%
\if@filesw \immediate \write \@auxout {\string \citation {#2}}\fi
\@tempcntb\m@ne \let\@h@ld\relax \def\@citea{}%
\@min\m@ne%
\@cite{%
  \@for \@citeb:=#2\do {\@ifundefined {b@\@citeb}%
    {\@h@ld\@citea\@tempcntb\m@ne\textbf{?}%% {\bf ?}
    \@warning {Citation `\@citeb ' on page \thepage \space undefined}}%
{\@minsofar\z@ \@for \@scan@cites:=#2\do {%
  \@ifundefined{b@\@scan@cites}%
    {\@cite@temp\m@ne}
    {\@cite@temp\number\csname b@\@scan@cites \endcsname \relax}%% <
 \ifnum\@cite@temp > \@min% select the next one to list
    \ifnum\@minsofar = \z@
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\else
    \ifnum\@cite@temp < \@minsofar %% >
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\fi\fi\fi}\@tempcnta\@min %% <
  \ifnum\@minsofar > \z@ % some more
    \advance\@tempcnta\@ne
    \@min\@minsofar
    \ifnum\@tempcnta=\@minsofar %   Number follows previous--hold on to it
      \ifx\@h@ld\relax
        \edef \@h@ld{\@citea\csname b@\@scan@copy\endcsname}%
      \else %% [
       \edef\@h@ld{\ifmmode{\mbox{--}\allowbreak }%
        \else --\allowbreak \fi %% ]
         \csname b@\@scan@copy\endcsname}%
      \fi
    \else \@h@ld\@citea\csname b@\@scan@copy\endcsname
          \let\@h@ld\relax
  \fi % no more
 \fi}%
  \def\@citea{%% [
   ,\,\allowbreak}}\@h@ld}{#1}% ]
}

%%%==========================================================

\def\thebibliography#1{\subsection*{References}\list
    {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
    \advance\leftmargin\labelsep \itemsep 0pt plus .5pt
    \usecounter{enumi}}
    \def\newblock{\hskip .11em plus .33em minus .07em}
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

%% appendix command for one single appendix
\def\appendix{\par
    \setcounter{section}{0}\setcounter{subsection}{0}
    \def\thesection{\Alph{section}} \section*{Appendix}
}
%% appendices command for multiple appendicies
\def\appendices{\par
    \setcounter{section}{0}\setcounter{subsection}{0}
    \def\thesection{\Alph{section}} \section*{Appendices}
}

\def\maketitle{\par
  \begingroup
  \def\thefootnote{} % the \thanks{} mark type is empty
  \if@twocolumn
     \if@technote
        \newpage\global\@topnum\z@ \@maketitle
     \else\twocolumn[\@maketitle]\fi
  \else\newpage\global\@topnum\z@ \@maketitle \fi
  \thispagestyle{ECTI} \@received \@thanks
  \endgroup
  \setcounter{footnote}{0}
  \let\maketitle\relax
  \let\@maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\thanks\relax
  \if@draftversion \vfil\mbox{}\newpage\fi
}

\def\mm@yy#1#2{%
\ifcase #2\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\else
   xxx\protect\ECTI@warning{check \string#1 command}\fi
}%

\def\received#1#2#3#4#5#6{\begingroup
\protected@xdef\@received{\protect\footnotetext[0]{%
  \hspace{2mm}Manuscript received on \mm@yy{\received}{#2} #3, #1 ; revised on \mm@yy{\received}{#5} #6, #4.}}\endgroup}
\def\@received{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             AUTHOR, MEMBERSHIP, AFFILIATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newtoks\@authorlist
\newcount\@authorcnt
\newcount\total@mbr
\newcount\@alcnt
\newcount\p@alcnt
\p@alcnt 6\relax

\def\authorlist#1{%
\@authorlist={#1}%
\@chkauthormembership }

\def\authorentry{\@ifnextchar[%% ]
{\E@ae}{\NE@ae}}
\def\NE@ae#1#2#3{\@ifnextchar[%% ]
  {\NE@aeP{#1}{#2}{#3}}%   no Email with present aff.
  {\NE@aeNP{#1}{#2}{#3}}}% no Email without present aff.
\def\E@ae[#1]#2#3#4{\@ifnextchar[%% ]
  {\E@aeP[#1]{#2}{#3}{#4}}%    Email with present aff.
  {\E@aeNP[#1]{#2}{#3}{#4}}}%  Email without present aff.

\def\NE@aeNP#1#2#3{% no Email without present aff.
\@tempcnta\z@
\advance \@authorcnt \@ne
  \get@al{#3}%
  \def\al@none{0}%
  \fmt@an{#1}%
  \expandafter\xdef\csname @authorname\number\@authorcnt\endcsname{%
   \@authorname $^{#3}$}%
  \fmt@Ran{#1}%
  \expandafter\xdef
   \csname @authormembership\number\@authorcnt\endcsname{%
    \csname #2member\endcsname}%
  \expandafter\xdef
   \csname @authormembership\number\@authorcnt*\endcsname{%
    \csname #2members\endcsname}%
  \global\total@mbr=\@authorcnt
}

\def\get@al#1{%
\@for\get@label:=#1\do{%
  \@ifundefined{al@\get@label}%
   {\def\@noaffi{none}\def\@tempaffi{#1}%
    \ifx\@noaffi\@tempaffi\else
     \global\advance \@alcnt \@ne\fi
    \expandafter\xdef\csname al@\get@label\endcsname{\the \@alcnt}%
    \advance \@tempcnta \@ne
    \expandafter\xdef\csname al@\romannumeral\@tempcnta\endcsname{%
     \get@label}%
   }%
   {\advance \@tempcnta \@ne
    \expandafter\xdef\csname al@\romannumeral\@tempcnta\endcsname{%
     \get@label}%
   }%
}%
}

\def\fmt@an#1{%
\@temptokena{#1}%
\setbox\@tempboxa\hbox{\spaceskip=100mm #1}%
\ifdim \wd\@tempboxa < 100mm %% >
  \ieice@warning{a space should be added between 1st and 2nd name: "#1"}%
\else
  \def\name@form##1 ##2\@nil{\gdef\@tempan{\Large \textbf{##1} {\Large \textbf{##2}}}}%
  \expandafter\name@form\the\@temptokena\@nil
  \expandafter\@temptokena\expandafter{\@tempan}%
\fi
\def\@authorname{\the\@temptokena}%
}

\def\fmt@Ran#1{%
\ifnum\@authorcnt<\thr@@ %% >
  \expandafter\newtoks\csname @toks\romannumeral\@authorcnt\endcsname
  \csname @toks\romannumeral\@authorcnt\endcsname{#1}%
  \expandafter\xdef\csname @Rauthorname\number\@authorcnt\endcsname{%
   \noexpand\the\csname @toks\romannumeral\@authorcnt\endcsname}%
\fi }

\newcount\chk@al@cnt
\newcount\chk@p@al@cnt
\chk@p@al@cnt 6\relax

\def\affiliate[#1]#2{%
\advance \chk@al@cnt \@ne
\@ifundefined{al@#1}{%
  \expandafter\def\csname al@#1\endcsname{1}%
  \ieice@warning{Check the "#1" label, it does not correspond to
    ^^J\space\space any labels in \string\authorentry\space command}%
}{}%
  \protected@xdef\@affiliate{\@affiliate
   \protect \indent $^{#1}${#2} \\ }%
}
\def\@affiliate{}

\def\ft@mailaddress#1{%
\protected@xdef\@mailaddress{%
\@mailaddress
\protect\footnotetext[0]{#1}}%
}
\def\@mailaddress{}

\def\p@affiliate{}

\def\mmember{Member}
\def\mmembers{Members}
\def\nmember{Non-member}
\def\nmembers{Non-members}
\def\smember{Student Member}
\def\smembers{Student Members}
\def\amember{Associate Member}
\def\amembers{Associate Members}
\def\hmember{Honorary Member}
\def\hmembers{Honorary Members}
\def\rmember{Regular Member}
\def\rmembers{Regular Members}
\def\fmember{Fellow}
\def\fmembers{Fellows}

\newcount\@authorcnt@p@ne
\newif\if@plural
\def\@chkauthormembership{\the\@authorlist
\@authorcnt\z@ \@pluralfalse
\loop\ifnum\@authorcnt<\total@mbr %% >
  \advance \@authorcnt \@ne
  \@authorcnt@p@ne=\@authorcnt \advance \@authorcnt@p@ne \@ne
  \if@plural
    \edef\@nextmembership{%
     \csname @authormembership\number\@authorcnt@p@ne*\endcsname}
  \else
    \edef\@nextmembership{%
     \csname @authormembership\number\@authorcnt@p@ne\endcsname}
  \fi
  \expandafter\ifx\csname @authormembership\number\@authorcnt\endcsname
   \@nextmembership
    \expandafter\gdef\csname @authormembership\number\@authorcnt\endcsname{%
     \relax}
    \edef\tempauthormembership{%
     \csname @authormembership\number\@authorcnt@p@ne*\endcsname}
    \expandafter\let
     \csname @authormembership\number\@authorcnt@p@ne\endcsname
       \tempauthormembership \@pluraltrue
  \else
   \@pluralfalse
  \fi
\repeat }

\newcount\totalnumberm@ne
\newdimen\authorlistlength
\authorlistlength=170mm

\def\breakauthorline#1{%
\@for\@breakauthor:=#1\do{%
  \expandafter\gdef\csname @breakauthorline\@breakauthor\endcsname%
}%
  \@authorcnt=\z@
  \loop\ifnum\@authorcnt<\total@mbr % >
   \advance \@authorcnt \@ne
    \@ifundefined{@breakauthorline\number\@authorcnt}%
     {\expandafter\gdef\csname
@breakauthorline\number\@authorcnt\endcsname{%
      \relax}}
     {\relax}%
  \repeat
}

\def\@outputauthorlist{%
\hfill \vbox{\hsize=\authorlistlength
  \baselineskip 3.75mm
  \centering
  \def\no@authormembership{\relax}%
  \@authorcnt\z@
  \totalnumberm@ne\total@mbr \advance \totalnumberm@ne -\@ne
  \loop\ifnum\@authorcnt<\total@mbr %% >
   \advance \@authorcnt \@ne
    \ifnum\@authorcnt < \totalnumberm@ne %% >
     \mbox{\csname @authorname\number\@authorcnt\endcsname
      \expandafter\ifx \csname @authormembership\number\@authorcnt\endcsname
       \no@authormembership
      \else
       \nobreak,
      \fi
     \nobreak{\csname @authormembership\number\@authorcnt\endcsname}}\nobreak,
      \csname @breakauthorline\number\@authorcnt\endcsname
    \else
     \ifnum\@authorcnt=\total@mbr
      \mbox{\csname @authorname\number\@authorcnt\endcsname
       \expandafter\ifx \csname
@authormembership\number\@authorcnt\endcsname
        \no@authormembership
       \else
        \nobreak,
       \fi
      \nobreak{\csname @authormembership\number\@authorcnt\endcsname}}
        \csname @breakauthorline\number\@authorcnt\endcsname
     \else
      \ifnum\total@mbr=\tw@
       \mbox{\csname @authorname\number\@authorcnt\endcsname
        \expandafter\ifx \csname
@authormembership\number\@authorcnt\endcsname
         \no@authormembership
        \else
         \nobreak,
        \fi
       \nobreak{\csname @authormembership\number\@authorcnt\endcsname}}%
        \csname @breakauthorline\number\@authorcnt\endcsname
       \mbox{\ and\/\ }\nobreak
      \else
       \mbox{\csname @authorname\number\@authorcnt\endcsname%
        \expandafter\ifx \csname
@authormembership\number\@authorcnt\endcsname
         \no@authormembership
        \else
         \nobreak,
        \fi
       \nobreak{\csname @authormembership\number\@authorcnt\endcsname},}%
       \csname @breakauthorline\number\@authorcnt\endcsname
       \mbox{\ and\/\ }\nobreak
      \fi
     \fi
    \fi
  \repeat
}%
\par\vskip 1.3em
\@affiliate
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% minor changes in \maketitle, \abstract and \keywords
%%% according to the ECTI format------ !PB
%   - Use \name instead of \author
%       - Use \address for authors' affiliations

\def\@maketitle{\newpage\if@draftversion\mbox{}\vfil\fi
        \begin{center}
    \if@technote
      {\bf\@title\par}\vskip 1.3em{\lineskip .5em\@author\par}
    \else
      \vskip-0.4em{\huge \bf \@title\par}\vskip1.5em%
      \if@draftversion\vfil\fi%
      {\lineskip 0.5em
      \@outputauthorlist}
    \fi
    \end{center}\par\vskip 1.3em}

\def\abstract{%
    \if@twocolumn
        \begin{flushleft}\normalsize{\bf{ABSTRACT}}\end{flushleft}\par
    \else
      \begin{center}\vspace{-0.8em}\small\bf ABSTRACT\end{center}\quotation\small
    \fi}
\def\endabstract{\vspace{0.1em}\par\if@twocolumn\else\endquotation\fi
    \normalsize\rm}

\def\keywords{%
        \vskip 1em
    \if@twocolumn
     \noindent \normalsize{\bf{Keywords}}: $\!$%
    \else
      \begin{center}\vspace{-0.8em}\small\bf Keywords\end{center}\quotation\small
    \fi}
\def\endkeywords{\vspace{0.1em}\par\if@twocolumn\else\endquotation\fi
    \normalsize\rm}

%% FIX numbering of sections.
%\gdef\@punct{.\ \ }  % Punctuation after run-in section heading
%\def\@sect#1#2#3#4#5#6[#7]#8{%
%  \ifnum #2>\c@secnumdepth
%     \def\@svsec{}
%  \else
%     \refstepcounter{#1}\edef\@svsec{%
%     \ifnum #2=1\Roman{#1}.\fi\ifnum #2=2\Alph{#1}.\fi
%     \ifnum #2>2{\csname the#1dis\endcsname}\if@technote.\fi\fi \hskip .5em}
%  \fi
%  \@tempskipa #5\relax
%  \ifdim \@tempskipa>\z@
%     \begingroup #6\relax
%       \@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M #8\par}
%     \endgroup
%     \csname #1mark\endcsname{#7}
%     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth\else
%          \protect\numberline{\csname the#1\endcsname}\fi#7}
%  \else
%     \def\@svsechd{#6\hskip #3\@svsec #8\@punct\csname #1mark\endcsname{#7}
%     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth \else
%          \protect\numberline{\csname the#1\endcsname}\fi#7}}
%  \fi
%  \@xsect{#5}}

%\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
%  \ifdim \@tempskipa>\z@
%     \begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M #5\par}\endgroup
%  \else \def\@svsechd{#4\hskip #1\relax #5\@punct}\fi
%  \@xsect{#3}}


%% SECTIONS
%%
\def\section{\@startsection {section}{1}{\z@}{3.0ex plus 0.5ex
     minus .2ex}{1.0ex plus .2ex} {\bf\uppercase}}  %%!PB
\def\subsection{\@startsection{subsection}{2}{\z@}{3.0ex plus .5ex
     minus .2ex}{1.0ex plus .2ex}{\bf}}
\if@technote
  \def\subsubsection{\@startsection{subsubsection}{3}{\z@}{2.0ex plus .5ex
     minus .1ex}{0pt}{\it}}
\else
  \def\subsubsection{\@startsection{subsubsection}{3}{\z@}{2.0ex plus .5ex
     minus .2ex}{1.0ex plus .2ex}{\rm}}
  \def\paragraph{\@startsection{paragraph}{4}{\z@}{1.0ex plus .5ex
     minus .2ex}{0ex}{\rm}}
\fi

%% ENVIRONEMTS
\def\QED{\mbox{\rule[0pt]{1.5ex}{1.5ex}}}
\def\proof{\noindent\hspace{2em}{\it Proof: }}
\def\endproof{\hspace*{\fill}~\QED\par\endtrivlist\unskip}
\def\@begintheorem#1#2{\tmpitemindent\itemindent\topsep 0pt\rm\trivlist
    \item[\hskip \labelsep{\indent\it #1\ #2:}]\itemindent\tmpitemindent}
\def\@opargbegintheorem#1#2#3{\tmpitemindent\itemindent\topsep 0pt\rm \trivlist
    \item[\hskip\labelsep{\indent\it #1\ #2\ \rm(#3)}]\itemindent\tmpitemindent}
\def\@endtheorem{\endtrivlist\unskip}

%% sets up default PAGESTYLE
\ps@ECTI
\pagenumbering{arabic}

\if@twoside\else\raggedbottom\fi
\if@tmptwocolumn \twocolumn \sloppy \flushbottom \fi
\if@draftversion \def\baselinestretch{1.5}\fi

%% Definition for Big letter at the beginning of a paragraph
%%
\def\PARstart#1#2{\begingroup\def\par{\endgraf\endgroup\lineskiplimit=0pt}
    \setbox2=\hbox{\uppercase{#2} }\newdimen\tmpht \tmpht \ht2
    \advance\tmpht by \baselineskip\font\hhuge=cmr10 at \tmpht
    \setbox1=\hbox{{\hhuge #1}}
    \count7=\tmpht \count8=\ht1\divide\count8 by 1000 \divide\count7 by\count8
    \tmpht=.001\tmpht\multiply\tmpht by \count7\font\hhuge=cmr10 at \tmpht
    \setbox1=\hbox{{\hhuge #1}} \noindent \hangindent1.05\wd1
    \hangafter=-2 {\hskip-\hangindent \lower1\ht1\hbox{\raise1.0\ht2\copy1}%
    \kern-0\wd1}\copy2\lineskiplimit=-1000pt}

\if@technote\def\PARstart#1#2{#1#2}\fi     % if technical note, disable it
\if@draftversion\def\PARstart#1#2{#1#2}\fi % if draft, disable it

%% Allows to enter BIOGRAPHY leaving place for picture (adapts to font size)
%%
\newenvironment{biography}[2]{%
\footnotesize\unitlength 1mm\bigskip\bigskip\bigskip\parskip=0pt\par%
\rule{0pt}{39mm}\vspace{-39mm}\par%   garantees correct page breaking
\noindent\setbox0\hbox{\includegraphics[width=1.05in,height=1.3in,clip]{#1}}% [width=1.05in,height=1.3in]  box containing the frame 
\ht0=37mm\count10=\ht0\divide\count10 by\baselineskip%  calculates lines
\global\hangindent29mm\global\hangafter-\count10%
\hskip-28.5mm\setbox0\hbox to 28.5mm {\raise-30.5mm\box0\hss}%
\dp0=0mm\ht0=0mm\box0\noindent\bf#2\rm}{
\par\rm\normalsize}


\endinput %!PN
