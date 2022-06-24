"""
https://www.tensorflow.org/tensorboard/dataframe_api?hl=ko
"""
from packaging import version
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
from bisect import bisect_left


# experiment_id = "4yT62ac3SBqokgGKiKBJMw" # hallway or lounge
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scalars(pivot=False)
# # df = df[0:100]  # computer psnr
#
#
# dfw = experiment.get_scalars(pivot=False)
# dfw_validation = dfw[dfw.run.str.endswith("arkit/hallway_1_01")]
# # Get the optimizer value for each row of the validation DataFrame.
# optimizer_validation = dfw_validation.run.apply(lambda run: run.split(",")[0]) #lambda run: run.split(",")[0]
#
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 1, 1)
# sns.lineplot(data=dfw_validation, x="step", y="value").set_title("PSNR") #hue=optimizer_validation
#
# dfw_validation = dfw[dfw.run.str.endswith("iphone/hallway_1_01")]
# # Get the optimizer value for each row of the validation DataFrame.
# optimizer_validation = dfw_validation.run.apply(lambda run: run.split(",")[0]) #lambda run: run.split(",")[0]
# sns.lineplot(data=dfw_validation, x="step", y="value").set_title("PSNR")#,hue=optimizer_validation
# # plt.subplot(1, 2, 2)
# # sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
# #              hue=optimizer_validation).set_title("loss")
#
# fname = "./pnsrplot.png"
# plt.savefig(fname, dpi=75)


#===============================================================================
experiment_id = "4yT62ac3SBqokgGKiKBJMw" # hallway or lounge
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars(pivot=False)
hall_arkit = df[df.run.str.endswith("arkit/hallway_1_01")]
hall_iphone = df[df.run.str.endswith("iphone/hallway_1_01")]

step=[i for i in range(0,201000,1000)]

hall_arkit_value=hall_arkit['value'].to_numpy()
hall_arkit_step=hall_arkit['step'].to_numpy()
hall_iphone_value=hall_iphone["value"].to_numpy()
hall_iphone_step=hall_iphone['step'].to_numpy()


def nearest_index(s,ts):
    # Given a presorted list of timestamps:  s = sorted(index)
    time_list = list(map(lambda t: abs(ts - t), s))
    return time_list.index(min(time_list))


plot_arkit_data = []
plot_iphone_data = []
for i in range(len(step)):
    index = nearest_index(hall_arkit_step , step[i])
    plot_arkit_data.append(hall_arkit_value[index])
    index = nearest_index(hall_iphone_step , step[i])
    plot_iphone_data.append(hall_iphone_value[index])

# df1 = pd.DataFrame([hall_arkit['value'],hall_iphone["value"]],columns=['Ours','BARF'])
fig = plt.figure(figsize=(18,10))
plt.plot(plot_arkit_data,'cx--', label='GT')
plt.plot(plot_iphone_data,'rx:', label='ours', markersize=5)
plt.title("Hallway PSNR")
plt.xlabel('step')
plt.ylabel('PSNR')
plt.legend(loc='best')
plt.show()

fname = "{}.png".format("Hallway")
plt.savefig(fname, dpi=75)
# clean up
plt.clf()

