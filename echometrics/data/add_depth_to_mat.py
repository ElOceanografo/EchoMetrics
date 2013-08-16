from scipy import io
import numpy as np

data = io.loadmat('MARS-D20100308-T234901.mat')
depth = np.linspace(max(data['Depth_start']),
                    min(data['Range_stop'] + data['Depth_start']),
                    max(data['Sample_count']))
data['Depth'] = depth
io.savemat('data/MARS-D20100308-T234901-depth.mat', data)