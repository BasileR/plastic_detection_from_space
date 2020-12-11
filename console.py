import utils
import spectral
import matplotlib.pyplot as plt
#utils.createSel('2019_04_18_M','B04')
#path = utils.create_plot_save_label('2019_04_18_M', ['B04'], zoom_int=10, save = True)
#path = utils.create_plot_save_label('2019_06_07_M', ['True_color'], zoom_int=10, save = True, index=False)

#print(path)


table, label = utils.get_batch([], '2019_06_07_M', all = True)
t = table.shape[2]
print(t)
plt.figure()
for i in range(1,t):
    plt.subplot(t,1,i)
    plt.imshow(table[:,:,i])
plt.show()

print(table.shape, label.shape)
