from scipy.stats import ttest_ind


cosine_tfidf = [
    0.0000, 0.0246, 0.3019, 0.0255, 0.0327, 0.1906,
    0.2171, 0.0163, 0.0665, 0.5059, 0.1562, 0.0000,
    0.0010, 0.0364, 0.1935, 0.0329, 0.0002, 0.0741,
    0.0159, 0.1804, 0.0009, 0.0075, 0.6044, 0.0064,
    0.0194, 0.0010, 0.1196, 0.0040, 0.2735, 0.2460,
    0.0230, 0.0012, 0.1141, 0.0207, 0.0024, 0.0038,
    0.0208, 0.0006, 0.0023, 0.0072, 0.5003, 0.0117,
    0.0051, 0.8960, 0.0082, 0.0000, 0.2851, 0.0000,
    0.0194, 0.0760]

bm25 = [
    0.0003, 0.1239, 0.7406, 0.0539, 0.1004, 0.2036,
    0.1129, 0.0771, 0.1516, 0.6045, 0.1375, 0.0104,
    0.0939, 0.1742, 0.2457, 0.2282, 0.1682, 0.0789,
    0.0578, 0.3085, 0.0189, 0.1250, 0.6263, 0.0386,
    0.1952, 0.0025, 0.1550, 0.0277, 0.4234, 0.5660,
    0.2207, 0.0064, 0.0027, 0.1252, 0.0121, 0.0433,
    0.0072, 0.0498, 0.0043, 0.0057, 0.4378, 0.0023,
    0.0225, 0.4678, 0.0912, 0.0084, 0.4698, 0.0116,
    0.0467, 0.1196]

bm25va = [
    0.0000, 0.1267, 0.6736, 0.0712, 0.1200, 0.2048,
    0.1294, 0.0810, 0.1344, 0.5308, 0.1526, 0.0169,
    0.0810, 0.1294, 0.2517, 0.2212, 0.2048, 0.0973,
    0.0580, 0.3126, 0.0402, 0.0997, 0.6713, 0.0227,
    0.1526, 0.0030, 0.1747, 0.0280, 0.3139, 0.5080,
    0.2460, 0.0097, 0.0119, 0.1203, 0.0156, 0.0336,
    0.0089, 0.0617, 0.0033, 0.0093, 0.4746, 0.0128,
    0.0281, 0.4112, 0.0653, 0.0155, 0.2540, 0.0181,
    0.0691, 0.0945]

equal_var = False

print(ttest_ind(cosine_tfidf, bm25, equal_var=equal_var))
print(ttest_ind(cosine_tfidf, bm25va, equal_var=equal_var))
print(ttest_ind(bm25, bm25va, equal_var=equal_var))