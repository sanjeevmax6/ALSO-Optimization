import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# not using 4 and 5
rmse_also = (0.23658282217307114+0.37924879557703145+0.8237369325923+0.843165806632246+1.0594933884985398+1.8679691316395726)/6
mae_also = (1.3437593009279107+1.7596636829809558+4.467487240479336+4.304332943112697+8.959818399687226+17.272412810434254)/6
mase_also = (0.055971431747375+0.143829648946629+0.6785425341165714+0.7109285774738062+1.122526240272118+3.4893086767582986)/6


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
plt.savefig('baselineComparison_memory.png')
plt.show()
