"""Reference list, keyed by citation tag.

Every entry was retrieved from CrossRef by ref_check.py / cr_meta and carries
the DOI that the API returned for it, so no citation here is written from
memory.  Titles follow the journal rule of capitalising only the first word
and proper nouns.  Run analysis/verify_refs.py after any edit: it re-checks
title, first author, year, volume, issue and pages field-by-field.

JETIA requires that at least half of the reference list be no older than
five years, that at least 70 per cent be international with respect to the
corresponding author's country, that at least 40 per cent be available
online, and that self-citations to JETIA stay below 10 per cent of the list.
analysis/ref_stats.py checks all four against the entries actually cited.
"""

REFS = {
    # ------------------------------------------------ news and asset prices
    "tetlock": 'P. C. Tetlock, "Giving content to investor sentiment: the '
               'role of media in the stock market," The Journal of Finance, '
               'vol. 62, no. 3, pp. 1139-1168, 2007. '
               'doi:10.1111/j.1540-6261.2007.01232.x',
    "lm2011": 'T. Loughran, B. McDonald, "When is a liability not a '
              'liability? Textual analysis, dictionaries, and 10-Ks," The '
              'Journal of Finance, vol. 66, no. 1, pp. 35-65, 2011. '
              'doi:10.1111/j.1540-6261.2010.01625.x',
    "lm2020": 'T. Loughran, B. McDonald, "Textual analysis in finance," '
              'Annual Review of Financial Economics, vol. 12, no. 1, '
              'pp. 357-375, 2020. doi:10.1146/annurev-financial-012820-032249',
    "tetlock2008": 'P. C. Tetlock, M. Saar-Tsechansky, S. Macskassy, "More '
                   'than words: quantifying language to measure firms\' '
                   'fundamentals," The Journal of Finance, vol. 63, no. 3, '
                   'pp. 1437-1467, 2008. '
                   'doi:10.1111/j.1540-6261.2008.01362.x',
    "todd": 'A. Todd, J. Bowden, Y. Moshfeghi, "Text-based sentiment '
            'analysis in finance: synthesising the existing literature and '
            'exploring future directions," Intelligent Systems in '
            'Accounting, Finance and Management, vol. 31, no. 1, e1549, '
            '2024. doi:10.1002/isaf.1549',
    "chan": 'W. S. Chan, "Stock price reaction to news and no-news: drift '
            'and reversal after headlines," Journal of Financial Economics, '
            'vol. 70, no. 2, pp. 223-260, 2003. '
            'doi:10.1016/S0304-405X(03)00146-6',
    "baker": 'S. R. Baker, N. Bloom, S. J. Davis, K. Kost, "Policy news and '
             'stock market volatility," Journal of Financial Economics, '
             'vol. 175, 104187, 2026. doi:10.1016/j.jfineco.2025.104187',

    # ------------------------------------------------ learned architectures
    "haryono": 'A. T. Haryono, R. Sarno, K. R. Sungkono, "Transformer-gated '
               'recurrent unit method for predicting stock price based on '
               'news sentiments and technical indicators," IEEE Access, '
               'vol. 11, pp. 77132-77146, 2023. '
               'doi:10.1109/ACCESS.2023.3298445',
    "mu": 'G. Mu, N. Gao, Y. Wang, L. Dai, "A stock price prediction model '
          'based on investor sentiment and optimized deep learning," IEEE '
          'Access, vol. 11, pp. 51353-51367, 2023. '
          'doi:10.1109/ACCESS.2023.3278790',
    "choi": 'J. Choi, S. Yoo, X. Zhou, Y. Kim, "Hybrid information mixing '
            'module for stock movement prediction," IEEE Access, vol. 11, '
            'pp. 28781-28790, 2023. doi:10.1109/ACCESS.2023.3258695',
    "ho": 'T. T. Ho, Y. Huang, "Stock price movement prediction using '
          'sentiment analysis and CandleStick chart representation," '
          'Sensors, vol. 21, no. 23, 7957, 2021. doi:10.3390/s21237957',
    "snasel": 'V. Snášel, J. D. Velasquez, M. Pant, D. Georgiou, L. Kong, '
              '"A generalization of multi-source fusion-based framework to '
              'stock selection," Information Fusion, vol. 102, 102018, '
              '2024. doi:10.1016/j.inffus.2023.102018',
    "long": 'W. Long, J. Gao, K. Bai, Z. Lu, "A hybrid model for stock '
            'price prediction based on multi-view heterogeneous data," '
            'Financial Innovation, vol. 10, no. 1, 2024. '
            'doi:10.1186/s40854-023-00519-w',
    "ozbayoglu": 'A. M. Ozbayoglu, M. U. Gudelek, O. B. Sezer, "Deep '
                 'learning for financial applications: a survey," Applied '
                 'Soft Computing, vol. 93, 106384, 2020. '
                 'doi:10.1016/j.asoc.2020.106384',
    "sezer": 'O. B. Sezer, M. U. Gudelek, A. M. Ozbayoglu, "Financial time '
             'series forecasting with deep learning: a systematic literature '
             'review: 2005-2019," Applied Soft Computing, vol. 90, 106181, '
             '2020. doi:10.1016/j.asoc.2020.106181',
    "xu": 'Y. Xu, S. B. Cohen, "Stock movement prediction from tweets and '
          'historical prices," in Proc. 56th Annual Meeting of the '
          'Association for Computational Linguistics, pp. 1970-1979, 2018. '
          'doi:10.18653/v1/P18-1183',
    "hu": 'Z. Hu, W. Liu, J. Bian, X. Liu, T. Y. Liu, "Listening to chaotic '
          'whispers: a deep learning framework for news-oriented stock '
          'trend prediction," in Proc. 11th ACM Int. Conf. Web Search and '
          'Data Mining, pp. 261-269, 2018. doi:10.1145/3159652.3159690',
    "sawhney": 'R. Sawhney, S. Agarwal, A. Wadhwa, R. R. Shah, "Deep '
               'attentive learning for stock movement prediction from social '
               'media text and company correlations," in Proc. 2020 Conf. '
               'Empirical Methods in Natural Language Processing, '
               'pp. 8415-8426, 2020. doi:10.18653/v1/2020.emnlp-main.676',
    "patel": 'J. Patel, S. Shah, P. Thakkar, K. Kotecha, "Predicting stock '
             'market index using fusion of machine learning techniques," '
             'Expert Systems with Applications, vol. 42, no. 4, '
             'pp. 2162-2172, 2015. doi:10.1016/j.eswa.2014.10.031',
    "bao": 'W. Bao, J. Yue, Y. Rao, "A deep learning framework for financial '
           'time series using stacked autoencoders and long-short term '
           'memory," PLOS ONE, vol. 12, no. 7, e0180944, 2017. '
           'doi:10.1371/journal.pone.0180944',
    "zhangtrans": 'Q. Zhang et al., "Transformer-based attention network for '
                  'stock movement prediction," Expert Systems with '
                  'Applications, vol. 202, 117239, 2022. '
                  'doi:10.1016/j.eswa.2022.117239',
    "ligan": 'S. Li, S. Xu, "Enhancing stock price prediction using GANs and '
             'transformer-based attention mechanisms," Empirical Economics, '
             'vol. 68, no. 1, pp. 373-403, 2025. '
             'doi:10.1007/s00181-024-02644-6',
    "mintarya": 'L. Mintarya, J. Halim, C. Angie, S. Achmad, A. Kurniawan, '
                '"Machine learning approaches in stock market prediction: a '
                'systematic literature review," Procedia Computer Science, '
                'vol. 216, pp. 96-102, 2023. doi:10.1016/j.procs.2022.12.115',
    "saberi": 'M. Saberironaghi, J. Ren, A. Saberironaghi, "Stock market '
              'prediction using machine learning and deep learning '
              'techniques: a review," AppliedMath, vol. 5, no. 3, 76, 2025. '
              'doi:10.3390/appliedmath5030076',

    # ------------------------------------------------ scorers and lexica
    "finbert": 'Z. Liu, D. Huang, K. Huang, Z. Li, J. Zhao, "FinBERT: a '
               'pre-trained financial language representation model for '
               'financial text mining," in Proc. 29th Int. Joint Conf. '
               'Artificial Intelligence, pp. 4513-4519, 2020. '
               'doi:10.24963/ijcai.2020/622',
    "bert": 'J. Devlin, M. W. Chang, K. Lee, K. Toutanova, "BERT: '
            'pre-training of deep bidirectional transformers for language '
            'understanding," in Proc. 2019 Conf. North American Chapter of '
            'the Association for Computational Linguistics, pp. 4171-4186, '
            '2019. doi:10.18653/v1/N19-1423',
    "vader": 'C. Hutto, E. Gilbert, "VADER: a parsimonious rule-based model '
             'for sentiment analysis of social media text," in Proc. Int. '
             'AAAI Conf. Web and Social Media, vol. 8, no. 1, pp. 216-225, '
             '2014. doi:10.1609/icwsm.v8i1.14550',
    "consoli": 'S. Consoli, L. Barbaglia, S. Manzan, "Fine-grained, '
               'aspect-based sentiment analysis on economic and financial '
               'lexicon," Knowledge-Based Systems, vol. 247, 108781, 2022. '
               'doi:10.1016/j.knosys.2022.108781',
    "linlex": 'W. Lin, L. Liao, "Lexicon-based prompt for financial '
              'dimensional sentiment analysis," Expert Systems with '
              'Applications, vol. 244, 122936, 2024. '
              'doi:10.1016/j.eswa.2023.122936',
    "omojowo": 'F. Omojowo, "Comparative evaluation of lexicon-based and '
               'transformer-based sentiment analysis tools," Discover '
               'Computing, vol. 29, no. 1, 181, 2026. '
               'doi:10.1007/s10791-026-09967-1',
    "ruan": 'L. Ruan, H. Jiang, "Stock price prediction using '
            'FinBERT-enhanced sentiment with SHAP explainability and '
            'differential privacy," Mathematics, vol. 13, no. 17, 2747, '
            '2025. doi:10.3390/math13172747',
    "shao": 'Z. Shao, X. Yao, F. Chen, Z. Wang, J. Gao, "Revisiting '
            'time-varying dynamics in stock market forecasting: a '
            'multi-source sentiment analysis approach with large language '
            'model," Decision Support Systems, vol. 190, 114362, 2025. '
            'doi:10.1016/j.dss.2024.114362',
    "chenllm": 'W. Chen, W. Liu, J. Zheng, X. Zhang, "Leveraging large '
               'language model as news sentiment predictor in stock markets: '
               'a knowledge-enhanced strategy," Discover Computing, vol. 28, '
               'no. 1, 74, 2025. doi:10.1007/s10791-025-09573-7',

    # -------------------------------------------- sentiment and volatility
    "parkinson": 'M. Parkinson, "The extreme value method for estimating the '
                 'variance of the rate of return," The Journal of Business, '
                 'vol. 53, no. 1, pp. 61-65, 1980. doi:10.1086/296071',
    "engle": 'R. F. Engle, V. K. Ng, "Measuring and testing the impact of '
             'news on volatility," The Journal of Finance, vol. 48, no. 5, '
             'pp. 1749-1778, 1993. doi:10.1111/j.1540-6261.1993.tb05127.x',
    "linvol": 'P. Lin, S. Ma, R. Fildes, "The extra value of online investor '
              'sentiment measures on forecasting stock return volatility: a '
              'large-scale longitudinal evaluation based on Chinese stock '
              'market," Expert Systems with Applications, vol. 238, 121927, '
              '2024. doi:10.1016/j.eswa.2023.121927',
    "lei": 'B. Lei, Y. Song, "Volatility forecasting for stock market '
           'incorporating media reports, investors\' sentiment, and '
           'attention based on MTGNN model," Journal of Forecasting, '
           'vol. 43, no. 5, pp. 1706-1730, 2024. doi:10.1002/for.3101',
    "saravanos": 'C. Saravanos, A. Kanavos, "Forecasting stock market '
                 'volatility using social media sentiment analysis," Neural '
                 'Computing and Applications, vol. 37, no. 17, '
                 'pp. 10771-10794, 2025. doi:10.1007/s00521-024-10807-w',
    "zhanggnn": 'P. Zhang, R. Harris, J. Zheng, "GNN-based social media '
                'sentiment analysis for stock market forecasting and '
                'trading," Expert Systems with Applications, vol. 291, '
                '128425, 2025. doi:10.1016/j.eswa.2025.128425',
    "fernandes": 'M. Fernandes, M. Pereira, "Forecasting realized volatility '
                 'using news flow," The Quarterly Review of Economics and '
                 'Finance, vol. 104, 102040, 2025. '
                 'doi:10.1016/j.qref.2025.102040',
    "feng": 'Y. Feng, Y. Zhang, "Forecasting realized volatility: the choice '
            'of window size," Journal of Forecasting, vol. 44, no. 2, '
            'pp. 692-705, 2025. doi:10.1002/for.3221',

    # ------------------------------------------------ measurement error
    "fuller": 'W. A. Fuller, Measurement error models. Wiley Series in '
              'Probability and Statistics, 1987. doi:10.1002/9780470316665',
    "carroll": 'R. J. Carroll, D. Ruppert, L. A. Stefanski, C. M. '
               'Crainiceanu, Measurement error in nonlinear models: a modern '
               'perspective, 2nd ed. Chapman and Hall/CRC, 2006. '
               'doi:10.1201/9781420010138',
    "goldberger": 'A. S. Goldberger, "Structural equation methods in the '
                  'social sciences," Econometrica, vol. 40, no. 6, '
                  'pp. 979-1001, 1972. doi:10.2307/1913851',
    "griliches": 'Z. Griliches, "Distributed lags: a survey," Econometrica, '
                 'vol. 35, no. 1, pp. 16-49, 1967. doi:10.2307/1909382',
    "klauenberg": 'K. Klauenberg et al., "The GUM perspective on '
                  'straight-line errors-in-variables regression," '
                  'Measurement, vol. 187, 110340, 2022. '
                  'doi:10.1016/j.measurement.2021.110340',
    "qiao": 'M. Qiao, K. Huang, "Correcting measurement error in regression '
            'models with variables constructed from aggregated output of '
            'data mining models," MIS Quarterly, vol. 49, no. 1, pp. 29-60, '
            '2025. doi:10.25300/misq/2024/18026',
    "hayes": 'A. F. Hayes, P. D. Allison, S. Alexander, "Errors-in-variables '
             'regression as a viable approach to mediation analysis with '
             'random error-tainted measurements: estimation, effectiveness, '
             'and an easy-to-use implementation," Behavior Research Methods, '
             'vol. 57, no. 12, 323, 2025. doi:10.3758/s13428-025-02783-3',

    # ------------------------------------------------ filtering
    "turin": 'G. Turin, "An introduction to matched filters," IRE '
             'Transactions on Information Theory, vol. 6, no. 3, '
             'pp. 311-329, 1960. doi:10.1109/TIT.1960.1057571',
    "north": 'D. O. North, "An analysis of the factors which determine '
             'signal/noise discrimination in pulsed-carrier systems," '
             'Proceedings of the IEEE, vol. 51, no. 7, pp. 1016-1027, 1963. '
             'doi:10.1109/PROC.1963.2383',
    "widrow76": 'B. Widrow, J. McCool, M. Larimore, C. Johnson, "Stationary '
                'and nonstationary learning characteristics of the LMS '
                'adaptive filter," Proceedings of the IEEE, vol. 64, no. 8, '
                'pp. 1151-1162, 1976. doi:10.1109/PROC.1976.10286',
    "widrow84": 'B. Widrow, E. Walach, "On the statistical efficiency of the '
                'LMS algorithm with nonstationary inputs," IEEE Transactions '
                'on Information Theory, vol. 30, no. 2, pp. 211-221, 1984. '
                'doi:10.1109/TIT.1984.1056892',
    "slock": 'D. T. M. Slock, "On the convergence behavior of the LMS and '
             'the normalized LMS algorithms," IEEE Transactions on Signal '
             'Processing, vol. 41, no. 9, pp. 2811-2825, 1993. '
             'doi:10.1109/78.236504',
    "sayed": 'A. H. Sayed, Adaptive filters. Wiley-IEEE Press, 2008. '
             'doi:10.1002/9780470374122',
    "wangga": 'W. Wang, K. Dogancay, "Transient performance analysis of '
              'geometric algebra least mean square adaptive filter," IEEE '
              'Transactions on Circuits and Systems II: Express Briefs, '
              'vol. 68, no. 8, pp. 3027-3031, 2021. '
              'doi:10.1109/TCSII.2021.3069390',
    "eweda": 'E. Eweda, N. J. Bershad, J. C. M. Bermudez, "Stochastic '
             'analysis of the diffusion least mean square and normalized '
             'least mean square algorithms for cyclostationary white '
             'Gaussian and non-Gaussian inputs," International Journal of '
             'Adaptive Control and Signal Processing, vol. 35, no. 12, '
             'pp. 2466-2486, 2021. doi:10.1002/acs.3334',
    "linpmf": 'J. Lin, C. Jiang, P. Liu, "Parametric matched filter based on '
              'interference iteration," Digital Signal Processing, vol. 111, '
              '102962, 2021. doi:10.1016/j.dsp.2021.102962',
    "lincg": 'J. Lin, C. Jiang, J. Jiang, J. Kang, "Conjugate gradient '
             'persymmetric adaptive matched filter," Digital Signal '
             'Processing, vol. 123, 103395, 2022. '
             'doi:10.1016/j.dsp.2022.103395',
    "marcantoni": 'I. Marcantoni, A. Sbrollini, M. Morettini, C. A. Swenne, '
                  'L. Burattini, "Enhanced adaptive matched filter for '
                  'automated identification and measurement of '
                  'electrocardiographic alternans," Biomedical Signal '
                  'Processing and Control, vol. 68, 102619, 2021. '
                  'doi:10.1016/j.bspc.2021.102619',

    # ------------------------------------------------ data and inference
    "fnspid": 'Z. Dong, X. Fan, Z. Peng, "FNSPID: a comprehensive financial '
              'news dataset in time series," in Proc. 30th ACM SIGKDD Conf. '
              'Knowledge Discovery and Data Mining, pp. 4918-4927, 2024. '
              'doi:10.1145/3637528.3671629',
    "dm": 'F. X. Diebold, R. S. Mariano, "Comparing predictive accuracy," '
          'Journal of Business and Economic Statistics, vol. 13, no. 3, '
          'pp. 253-263, 1995. doi:10.1080/07350015.1995.10524599',
    "newey": 'W. K. Newey, K. D. West, "A simple, positive semi-definite, '
             'heteroskedasticity and autocorrelation consistent covariance '
             'matrix," Econometrica, vol. 55, no. 3, pp. 703-708, 1987. '
             'doi:10.2307/1913610',

    # ------------------------------------------------ this journal
    "jetia_finbert": 'N. Adelakun, A. Adebisi, "Sentiment analysis of '
                     'financial news using the BERT model," ITEGAM-JETIA, '
                     'vol. 10, no. 48, pp. 21-27, 2024. '
                     'doi:10.5935/jetia.v10i48.1029',
    "jetia_lstm": 'R. Nichani, L. Gasmi, S. Kabou, N. Laiche, "Novel '
                  'insights on the comparative study between LSTM and '
                  'transformer models for financial time series '
                  'prediction," ITEGAM-JETIA, vol. 11, no. 55, '
                  'pp. 212-221, 2025. doi:10.5935/jetia.v11i55.2658',
    "jetia_btc": 'A. Othman, S. Al-Banna Ali, M. Youssef, "Forecasting '
                 'Bitcoin price movements using historical data," '
                 'ITEGAM-JETIA, vol. 12, no. 60, pp. 584-595, 2026. '
                 'doi:10.5935/jetia.v12i60.3915',
    "jetia_fir": 'R. Amrane, "Curve-fitting-based FIR filter design for '
                 'stop-band attenuation performance improvement," '
                 'ITEGAM-JETIA, vol. 12, no. 59, pp. 773-779, 2026. '
                 'doi:10.5935/jetia.v12i59.3644',
}

# Publication year of each entry, used by ref_stats.py to check JETIA's
# "at least 50 per cent from the last five years" rule.  Kept next to the
# entries so the two cannot drift apart.
YEARS = {
    "tetlock": 2007, "lm2011": 2011, "lm2020": 2020, "tetlock2008": 2008,
    "todd": 2024, "chan": 2003, "baker": 2026, "haryono": 2023, "mu": 2023,
    "choi": 2023, "ho": 2021, "snasel": 2024, "long": 2024,
    "ozbayoglu": 2020, "sezer": 2020, "xu": 2018, "hu": 2018,
    "sawhney": 2020, "patel": 2015, "bao": 2017, "zhangtrans": 2022,
    "ligan": 2025, "mintarya": 2023, "saberi": 2025, "finbert": 2020,
    "bert": 2019, "vader": 2014, "consoli": 2022, "linlex": 2024,
    "omojowo": 2026, "ruan": 2025, "shao": 2025, "chenllm": 2025,
    "parkinson": 1980, "engle": 1993, "linvol": 2024, "lei": 2024,
    "saravanos": 2025, "zhanggnn": 2025, "fernandes": 2025, "feng": 2025,
    "fuller": 1987, "carroll": 2006, "goldberger": 1972, "griliches": 1967,
    "klauenberg": 2022, "qiao": 2025, "hayes": 2025, "turin": 1960,
    "north": 1963, "widrow76": 1976, "widrow84": 1984, "slock": 1993,
    "sayed": 2008, "wangga": 2021, "eweda": 2021, "linpmf": 2021,
    "lincg": 2022, "marcantoni": 2021, "fnspid": 2024, "dm": 1995,
    "newey": 1987, "jetia_finbert": 2024, "jetia_lstm": 2025,
    "jetia_btc": 2026, "jetia_fir": 2026,
}
