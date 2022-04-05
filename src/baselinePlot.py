from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# not using 4 and 5
rmse_also = (0.3864222673274799+0.7978789518047722+0.43635734353415206+0.5473523672040927+1.0031634962820701+0.991415682653361)/6
mae_also = (2.967185773998015+7.10061786949878+2.1345965985477218+2.849974709822229+8.162048691411966+7.2309366715722305)/6
mase_also = (0.14932216868651035+0.6366108217330819+0.19040773125618202+0.299594613883924+1.006337000272867+0.9829050558110298)/6


Methods = ["Proposed ALSO", "AR", "MA", "ARMA", "ARIMA", "ES"]
RMSE = [(100-rmse_also), 27.747, 28.052, 27.281, 26.942, 30.985]
MAE = [(100-mae_also), 23.514, 23.439, 23.219, 22.940, 27.085]
MASE = [mase_also, 0.778, 0.776, 0.769, 0.759, 0.897]

# X_axis = numpy.arrange(len(Methods))
X_axis = np.array([0, 1, 2, 3, 4, 5])

fig, ax = plt.subplots()

pps = [ax.bar(X_axis - 0.3, RMSE, 0.3, label = 'RMSE', color = '#7dcfb6'),
ax.bar(X_axis, MAE, 0.3, label = 'MAE', color = '#1d4e89'),
ax.bar(X_axis + 0.3, MASE, 0.3, label = 'MASE', color = '#f79256')]

ax.set_xticks(np.array([0, 1, 2, 3, 4, 5]))
ax.set_xticklabels(Methods)
ax.set_title("Baseline Comparsion Graphs")
ax.set_xlabel("Basline methods")
# ax.grid('on')
ax.set_ylabel("Accuracy percentage")
ax.bar_label(pps[0], padding=3)
ax.bar_label(pps[1], padding=3)
ax.bar_label(pps[2], padding=3)
ax.legend()
fig.tight_layout()

# for elem in pps:
#     for p in elem:
#         height = p.get_height()
#         ax.annotate('{:.2f}'.format(height),
#            xy=(p.get_x() + p.get_width() / 2, height),
#            xytext=(0, 5), # 3 points vertical offset
#            textcoords="offset points",
#            ha='center', va='bottom')
plt.savefig('baselineComparison_CPU.png')
plt.show()
