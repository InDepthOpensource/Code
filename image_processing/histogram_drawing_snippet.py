import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import matplotlib

if __name__ == '__main__':

    y = np.array([0.8870307390079942, 0.03736365804208509, 0.02173956131078224, 0.01377600897694239, 0.008356758101545982, 0.006609836441265856, 0.005079492806884249, 0.004464882977669133, 0.0034079811789772727, 0.002301399601942389, 0.0017414762404201903, 0.0013814591454149048, 0.0012021990205470401, 0.0010178547832980973, 0.0007424352536997886, 0.0005668656266516913, 0.0005870472301136364, 0.0005371867980311839, 0.00042716867071881606, 0.00036440440010570825, 0.0003312415350819239, 0.0003012272066596194, 0.00025288942752378434, 0.0001703822839587738, 9.030106203752643e-05, 7.014526625264271e-05, 5.0221739561310784e-05, 1.7936335557610993e-05, 9.316571419133192e-06, 7.8971491807611e-06, 2.580767706131078e-08, 0.0, 0.0, 0.0, 0.0])
    x = np.array([i * 0.5 for i in range(y.size)])

    # x = x * 100
    # y = y * 100

    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    # plt.grid(linestyle='--')

    ax.bar(x, y, width=0.3, align='edge', color='red')

    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel('m')
    plt.ylabel('Probability Density')

    fig.tight_layout()
    plt.show()
