'''
Segio:
https://github.com/equinor/segyio

Tutorial :
https://github.com/seg/tutorials-2015/blob/master/1512_Semblance_coherence_and_discontinuity/Discontinuity_tutorial.ipynb
'''
import numpy as np
import matplotlib.pyplot as plt
import segyio
from multiprocessing import Process
import scipy.ndimage
import scipy.signal
from functools import *

#from mayavi import mlab

class Seismic:
    def __init__(self,name = 'data/ST8511r92.segy'):
        with segyio.open( name,iline = 5, xline = 21) as f :
            data = list(f.trace.raw[:])
            i_len = np.size(f.ilines) #Inferred inline numbers
            x_len = np.size(f.xlines) #Inferred cross-line numbers
            d_len = np.size(f.samples) #Inferred sample offsets (frequency and recording time delay)
            #self.d = segyio.tools.cube(f)
        self.data = np.array(data).reshape((i_len,x_len,d_len))

    def get_color(MAX,MIN,DELT,num):
        u_delt = (abs(MAX - num)/DELT)*510
        l_delt = (abs(MIN - num)/DELT)*510
        col = (lambda : [l_delt/255,l_delt/255,1] if u_delt > 255.0 else [1,u_delt/255,u_delt/255])()
        return col
    
    def get_colors(lis):
        MAX = np.max(lis)
        MIN = np.min(lis)
        DELT = abs(MAX-MIN)
        lis = np.transpose(lis)[::-1].reshape((np.size(lis),))
        act = partial(Seismic.get_color,MAX,MIN,DELT)
        return [act(num) for num in lis]

    def draw_slice(data,name):
        w,d = np.shape(data)
        X,Y  = np.meshgrid(np.linspace(0,w-1,w), np.linspace(0,d-1,d)) 
        fig, ax = plt.subplots()
        ax.scatter(X,Y,marker='.',c = Seismic.get_colors(data))
        fig.suptitle(name)
        plt.show()
    
    def draw_pic(name,data,funct):
        data = funct(data)
        Seismic.draw_slice(data[0],name)
    
    def traces_print(name,data,st,dur,delt):
        w,h,d = data.shape
        data = data.reshape((w*h,d))
        si = len(data[0])
        data = np.append(np.zeros((1,si)),data,axis = 0)
        d_max = [np.max(data[st])]
        def m_f(i):
            d_max.append(np.max(data[i-1])+d_max[-1])
            return d_max[-1] * delt
        xs = [data[i]+m_f(i) for i in range(st+1,dur+1)]
        ys = [[i for i in range(si-1,-1,-1)] for u in range(dur)]
        fig, ax = plt.subplots()
        [ax.plot(xs[i], ys[i]) for i in range(dur)]
        fig.suptitle(name)
        plt.grid(True)
        plt.show()
        
    def initial(data):
        return data
    
    def bahorich_coherence(zwin,data):
        ni, nj, nk = data.shape
        out = np.zeros_like(data)
        # Pad the input to make indexing simpler. We're not concerned about memory usage.
        # We'll handle the boundaries by "reflecting" the data at the edge.
        padded = np.pad(data, ((0, 1), (0, 1), (zwin//2, zwin//2)), mode='reflect')

        for i, j, k in np.ndindex(ni, nj, nk):
            # Extract the "full" center trace
            center_trace = data[i,j,:]
        
            # Use a "moving window" portion of the adjacent traces
            x_trace = padded[i+1, j, k:k+zwin]
            y_trace = padded[i, j+1, k:k+zwin]

            # Cross correlate. `xcor` & `ycor` will be 1d arrays of length
            # `center_trace.size - x_trace.size + 1`
            xcor = np.correlate(center_trace, x_trace)
            ycor = np.correlate(center_trace, y_trace)
        
            # The result is the maximum normalized cross correlation value
            center_std = center_trace.std()
            px = xcor.max() / (xcor.size * center_std * x_trace.std())
            py = ycor.max() / (ycor.size * center_std * y_trace.std())
            out[i,j,k] = np.sqrt(px * py)
        return out
    
    def moving_window( window, func,data):
        # `generic_filter` will give the function 1D input. We'll reshape it for convinence
        wrapped = lambda region: func(region.reshape(window))
        # Instead of an explicit for loop, we'll use a scipy function to do the same thing
        # The boundaries will be handled by "reflecting" the data, by default
        return scipy.ndimage.generic_filter(data, wrapped, window)

    def marfurt_semblance(region):
        # We'll need an ntraces by nsamples array
        # This stacks all traces within the x-y "footprint" into one
        # two-dimensional array.
        region = region.reshape(-1, region.shape[-1])
        ntraces, nsamples = region.shape
        square_of_sums = np.sum(region, axis=0)**2
        sum_of_squares = np.sum(region**2, axis=0)
        sembl = square_of_sums.sum() / sum_of_squares.sum()
        return sembl / ntraces
    
    def gersztenkorn_eigenstructure(region):
        # Once again, stack all of the traces into one 2D array.
        region = region.reshape(-1, region.shape[-1])
        cov = region.dot(region.T)
        vals = np.linalg.eigvalsh(cov)
        return vals.max() / vals.sum()
    
    def gradients(seismic, sigma):
        #Builds a 4-d array of the gaussian gradient of *seismic*.
        grads = []
        for axis in range(3):
            # Gaussian filter with order=1 is a gaussian gradient operator
            grad = scipy.ndimage.gaussian_filter1d(seismic, sigma, axis=axis, order=1)
            grads.append(grad[..., np.newaxis])
        return np.concatenate(grads, axis=3)

    def moving_window4d(grad, window, func):
        """Applies the given function *func* over a moving *window*, reducing 
        the input *grad* array from 4D to 3D."""
        # Pad in the spatial dimensions, but leave the gradient dimension unpadded.
        half_window = [(x // 2, x // 2) for x in window] + [(0, 0)]
        padded = np.pad(grad, half_window, mode='reflect')
        out = np.empty(grad.shape[:3], dtype=float)
        for i, j, k in np.ndindex(out.shape):
            region = padded[i:i+window[0], j:j+window[1], k:k+window[2], :]
            out[i,j,k] = func(region)
        return out

    def gst_coherence_calc(region):
        """Calculate gradient structure tensor coherence on a local region.
        Intended to be applied with *moving_window4d*."""
        region = region.reshape(-1, 3)
        gst = region.T.dot(region) # This is the 3x3 gradient structure tensor
        # Reverse sort of eigenvalues of the GST (largest first)
        eigs = np.sort(np.linalg.eigvalsh(gst))[::-1]
        return (eigs[0] - eigs[1]) / (eigs[0] + eigs[1])
        
    def gst_coherence(window,sigma,seismic):
        """Randen et al's (2000) Gradient Structure Tensor based coherence."""
        # 4-d gradient array (ni x nj x nk x 3)
        grad = Seismic.gradients(seismic, sigma)
        return Seismic.moving_window4d(grad, window, Seismic.gst_coherence_calc)
    
    
    def explore3d(data_cube):
        source = mlab.pipeline.scalar_field(data_cube)
        source.spacing = [1, 1, -1]
        nx, ny, nz = data_cube.shape
        mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes', 
                                     slice_index=nx//2, colormap='gray')
        mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes', 
                                     slice_index=ny//2, colormap='gray')
        mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes', 
                                     slice_index=nz//2, colormap='gray')
        mlab.show()


    def main(self):
        Process(target = Seismic.traces_print,args = ("traces",self.data,0,301,1.3)).start()
        Process(target = Seismic.draw_pic,args = ("normal",self.data,Seismic.initial)).start()
        
        act = partial(Seismic.bahorich_coherence,21)
        Process(target = Seismic.draw_pic,args = ("Coherence",self.data[:1],act)).start()
        
        act = partial(Seismic.moving_window,(3, 3, 9),Seismic.marfurt_semblance)
        Process(target = Seismic.draw_pic,args = ("Semblance",self.data[0:2],act)).start()
        
        act = partial(Seismic.moving_window,(3, 3, 9),Seismic.gersztenkorn_eigenstructure)
        Process(target = Seismic.draw_pic,args = ("Eigenstructure",self.data[0:1],act)).start()
        
        act = partial(Seismic.gst_coherence,(3, 3, 9),1)
        Process(target = Seismic.draw_pic,args = ("Gradient Structure Tensor",self.data[0:2],act)).start()
        
        #new_data = Seismic.bahorich_coherence(21,self.data[:10])
        #Seismic.explore3d(new_data)
        
        return 0

if __name__ == "__main__":
    obj = Seismic()
    obj.main()

