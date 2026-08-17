"""Reference list, keyed by citation tag.

Every entry was retrieved from CrossRef by ref_check.py / ref_check2.py and
carries the DOI that the API returned for it, so no citation here is written
from memory.  Titles follow the journal rule of capitalising only the first
word and proper nouns.
"""

REFS = {
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
    "fnspid": 'Z. Dong, X. Fan, Z. Peng, "FNSPID: a comprehensive financial '
              'news dataset in time series," in Proc. 30th ACM SIGKDD Conf. '
              'Knowledge Discovery and Data Mining, pp. 4918-4927, 2024. '
              'doi:10.1145/3637528.3671629',
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
    "dm": 'F. X. Diebold, R. S. Mariano, "Comparing predictive accuracy," '
          'Journal of Business and Economic Statistics, vol. 13, no. 3, '
          'pp. 253-263, 1995. doi:10.1080/07350015.1995.10524599',
    "newey": 'W. K. Newey, K. D. West, "A simple, positive semi-definite, '
             'heteroskedasticity and autocorrelation consistent covariance '
             'matrix," Econometrica, vol. 55, no. 3, pp. 703-708, 1987. '
             'doi:10.2307/1913610',
    "aece_dga": 'X. H. Vu, X. D. Hoang, "An ensemble model for word-based '
                'DGA botnet detection using XGBoost and BERT," Advances in '
                'Electrical and Computer Engineering, vol. 25, no. 3, '
                'pp. 13-20, 2025. doi:10.4316/AECE.2025.03002',
    "aece_mpnn": 'G. Song, D. Fu, X. Wu, "A message passing neural network '
                 'framework with learnable PageRank for author impact '
                 'assessment," Advances in Electrical and Computer '
                 'Engineering, vol. 25, no. 1, pp. 11-20, 2025. '
                 'doi:10.4316/AECE.2025.01002',
    "aece_quant": 'M. Dubljanin, S. Panic, M. Savic, S. Milosavljevic, '
                  '"Adaptive µ-law gradient quantization for training '
                  'MLPs and CNNs," Advances in Electrical and Computer '
                  'Engineering, vol. 26, no. 1, pp. 55-64, 2026. '
                  'doi:10.4316/AECE.2026.01006',
    "patel": 'J. Patel, S. Shah, P. Thakkar, K. Kotecha, "Predicting stock '
             'market index using fusion of machine learning techniques," '
             'Expert Systems with Applications, vol. 42, no. 4, '
             'pp. 2162-2172, 2015. doi:10.1016/j.eswa.2014.10.031',
    "bao": 'W. Bao, J. Yue, Y. Rao, "A deep learning framework for financial '
           'time series using stacked autoencoders and long-short term '
           'memory," PLOS ONE, vol. 12, no. 7, e0180944, 2017. '
           'doi:10.1371/journal.pone.0180944',
    "parkinson": 'M. Parkinson, "The extreme value method for estimating the '
                 'variance of the rate of return," The Journal of Business, '
                 'vol. 53, no. 1, pp. 61-65, 1980. doi:10.1086/296071',
    "engle": 'R. F. Engle, V. K. Ng, "Measuring and testing the impact of '
             'news on volatility," The Journal of Finance, vol. 48, no. 5, '
             'pp. 1749-1778, 1993. doi:10.1111/j.1540-6261.1993.tb05127.x',
}
