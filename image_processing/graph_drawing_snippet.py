import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from scipy import interpolate

if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    y = [0.19136183646428884, 0.23198213734522524, 0.28056553868877526, 0.2948970923432339, 0.31857258756939205,
         0.32409270825879877, 0.34470713558127797, 0.3513088547551296, 0.36255160487310734, 0.3752991298827799,
         0.38330326320960534, 0.38731186763937886, 0.4027539524062707, 0.4169490852431043, 0.43463692582041863,
         0.4489276111656865, 0.45281586181875155, 0.4606964717944657, 0.47030497496236, 0.4664775603532289,
         0.464776408933437, 0.47133183631990233, 0.47276526996761914, 0.4844706034396439, 0.5029230923129535,
         0.520438139992128, 0.5314797713933329, 0.5410331558449526, 0.5485710435038985, 0.5524715457483448,
         0.5540201791421006, 0.560654457698292, 0.5665065091901209, 0.5618302196152511, 0.5591203802138703,
         0.5523446838814308, 0.5562885684311446, 0.56541303603641, 0.5759410684782604, 0.579005968250306,
         0.5820563087747523, 0.5934855877147389, 0.6010280453268771, 0.6063780564426932, 0.6094423610024109,
         0.6157396054371541, 0.619568118096671, 0.6279840833117046, 0.6313348402349781, 0.6398455120673363,
         0.6529969265739072, 0.6672957934959841, 0.6775691519839175, 0.6898938302409294, 0.7028085069153461,
         0.7068965643098764, 0.7058845628413604, 0.7118783496178673, 0.7089157338428147, 0.7219666434316596,
         0.7317206406022014, 0.7461392644360433, 0.7643494757683615, 0.7825789336512557, 0.8001217843757985,
         0.8092761431952692, 0.8209834440238606, 0.8235754464551304, 0.8203665128794783, 0.8159884904099429,
         0.8079755410643633, 0.7992875696382629, 0.794318940091263, 0.7924311272761916, 0.7799480675055607,
         0.7689996351960756, 0.7645544508303226, 0.7554462416017161, 0.7432661777387467, 0.7201953818954129,
         0.7007271480106586, 0.6736340107835117, 0.6528753048949771, 0.6406132728872657, 0.6288509264354617,
         0.6258515104373175, 0.6178125493914968, 0.6121655170803938, 0.587265445093822, 0.5709086330816338,
         0.566254089445024, 0.5717627787218219, 0.5633371651209713, 0.5432907924343877, 0.5172660095232137,
         0.509849012610717, 0.5153550706308213, 0.5064694665611439, 0.5017200328144127, 0.39200342534487603]

    # x = x[:-2]
    # y = y[:-2]

    xy = zip(x, y)
    xy = sorted(xy)

    x = [i[0] for i in xy]
    y = [i[1] for i in xy]

    x = np.array(x)
    y = (1 - np.array(y)) * 100

    #
    # x = np.array(x)
    # y = (1 - np.array(y)) * 100

    model = make_pipeline(PolynomialFeatures(degree=5, include_bias=True), LinearRegression())

    plt.rcParams.update({'font.size': 32})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "normal"

    # tck = interpolate.splrep(x, y, k=3, s=30)

    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    fig, ax = plt.subplots()
    plt.scatter(x[2:], y[2:], s=6 ** 2, color='tab:blue', clip_on=False)
    predicted = model.predict(x.reshape(-1, 1))
    plt.plot(x, predicted, linewidth=7, color='tab:orange')
    plt.grid(linestyle='-')

    print(list(x[2:]))
    print(list(y[2:]))

    x_range_start = 0
    x_range_end = 1.0

    y_range_start = 0
    y_range_end = 100

    x_range = np.linspace(x_range_start, x_range_end, 5, True)
    y_range = np.linspace(y_range_start, y_range_end, 5, True)
    plt.xlim([x_range_start - 0.03, x_range_end + 0.03])
    plt.ylim([y_range_start - 3, y_range_end + 3])

    ax.set_xticks(list(x_range))
    ax.set_yticks(list(y_range))
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('2')

    plt.xlabel(r'Relative illuminance')
    plt.ylabel(r'Missing depth (%)')
    plt.show()

# 559 * 414