import persim
from ripser import Rips
from persim import PersistenceImager
class set_up_rips():
  def __init__(self,data, maxdim = 1):
    self.data = data
    self.rips = Rips(maxdim = maxdim)
    self.dgms = self.rips.fit_transform(data)
  def get_rips_plot(self):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(self.data[:,0], self.data[:,1], s=4)
    plt.title("Scatter plot ")
    plt.subplot(122)
    self.rips.plot(self.dgms, legend=False, show=False)
    plt.title("Persistence diagram of $H_0$ and $H_1$")
    plt.show()
  def get_persist_dgm(self,dim =1):
    pimgr = PersistenceImager(pixel_size=1)
    pimgr.fit(self.dgms[dim])
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    pimgr.plot_diagram(self.dgms[1], skew=True, ax=axs[0])
    axs[0].set_title('Diagram', fontsize=16)
    pimgr.plot_image(pimgr.transform(self.dgms[1]), ax=axs[1])
    axs[1].set_title('Pixel Size: 1', fontsize=16)
    pimgr.pixel_size = 0.1
    pimgr.plot_image(pimgr.transform(self.dgms[1]), ax=axs[2])
    axs[2].set_title('Pixel Size: 0.1', fontsize=16)
    plt.tight_layout()
