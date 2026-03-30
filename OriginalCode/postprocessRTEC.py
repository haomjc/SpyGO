import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import struct
from general_utils import *
from matplotlib.colors import LightSource
from vispy.plot import Fig
# import easyPlot_myavi as ep

# from matplotlib import interactive
# interactive(true)

matplotlib.use('QtAgg')

def read_signals(file, all_steps = False):

    # import the csv with pandas as DataFrame
    data = pd.read_csv(file, skiprows=2, skipfooter=2, engine='python')
    step = np.array(data[['Step']]).squeeze()
    Fz = data[[' DAQ.Fz (N)']]
    COF = data[['DAQ.COF ()']]
    Fx = data[['DAQ.Fx (N)']]
    Zdepth = data[['XYZ.Z Depth (mm)']]
    try:
        vel = data[['Rotary.Velocity (rpm)']]
    except:
        vel = []

    t = np.array(data[[' Timestamp']]).squeeze()

    # find all the indices where the step number changes
    if all_steps == True:
        test = True
        stp = step
        val = 0
        counter = 0
        while test:
            stp -= stp[val]
            val = np.argwhere(stp>0)
            if not isempty(val):
                val = val[0][0]
                t[val:-1] += t[val-1]
                counter += 1
            else:
                test = False

    outputs = [COF, -Fx, Fz, Zdepth, vel]
    y_labels = ['COF (-)', '$F_x$ (N)', '$F_z$ (N)', '$z$ (mm)', '$\omega$ (rpm)']

    return outputs, t, y_labels

def plot_signal(signals, time, labels, index = 'all', title = 'Case XXX', fontsize = 18):
    matplotlib.rcParams.update({'font.size': fontsize})

    if not isinstance(index, str):
        signals = [signals[ii] for ii in index]
        labels = [labels[ii] for ii in index]
    N = len(signals)
    Ncell = 2
    rem = 0
    match N:
        case 1:
            Ncell = 1
            rem  = 0
        case 2:
            Ncell = 2
            rem = 1
        case 5:
            Ncell = 3
            rem = 1
    fig = plt.figure()
    for ii in range(0,N):
        ax = fig.add_subplot(Ncell-rem, Ncell, ii+1)
        ax.plot(time, signals[ii], linewidth = 0.1)
        ax.set_ylabel(labels[ii])
        ax.set_xlabel('$t$ (s)')
        ax.grid()

    fig.suptitle(title)
    return

def read_bcrf(file):
    skipbytes = 4096

    with open(file, 'rb') as f:
        f.seek(skipbytes)
        rawdata_bytes = f.read()
        # Unpack the raw data into float values
        format_string = '<' + 'f' * (len(rawdata_bytes) // struct.calcsize('f'))  # Assuming float32 format
        rawdata = np.array(list(struct.unpack(format_string, rawdata_bytes)))

    # Reopen the file to read the header and gather additional info
    with open(file, 'rb') as f:
        string_bytes = bytearray()
        header = []

        line_byte = f.read(4096) # read first 4096 bytes
        header = line_byte.decode('utf-16').replace('=', '\n').split('\n')
        # Initialize data structure
        datastruct = {}
        datastruct['xpixels'] = int(float(header[6]))
        datastruct['ypixels'] = int(float(header[8]))
        datastruct['zmin']    = float(header[10])*1e-3
        datastruct['xlen_mm'] = float(header[14])*1e-6
        datastruct['ylen_mm'] = float(header[16])*1e-6
        datastruct['xoff_mm'] = float(header[18])*1e-6
        datastruct['yoff_mm'] = float(header[20])*1e-6

    return rawdata, datastruct

def plot_3Dscan(Zdata, info, z_ratio = 0.3, crange = [580, 620], cross_section = False):
    xsize = info['xpixels']
    ysize = info['ypixels']
    xlen = info['xlen_mm']
    ylen = info['ylen_mm']
    Zdata = Zdata.reshape((ysize, xsize))
    x = np.linspace(-xlen/2, xlen/2, num = xsize)
    y = np.linspace(-ylen/2, ylen/2, num = ysize)
    X, Y = np.meshgrid(x, y)
    # F = ep.Figure() # init figure
    # ep.surface(F, X=x, Y = y, Z = Zdata/300)
    # F.start()
    import plotly.graph_objects as go
    objs=[go.Surface(z=Zdata, x=X, y=Y, colorscale="Reds", surfacecolor=Zdata, cmin = crange[0], cmax = crange[1],
        lighting=dict(
        ambient=0.2,       # Lower ambient to enhance shadows
        diffuse=0.6,       # Balanced diffuse for even lighting with some shadow
        specular=1.0,      # High specular to highlight small details
        roughness=0.1,     # Smooth surface for strong reflections
        fresnel=0.1        # Subtle Fresnel effect
        ),
        lightposition=dict(
            x=0,    # Adjust to create shadow angles that highlight details
            y=0,
            z=2     # Light source above and slightly off to one side
        ))]
    
    fig = go.Figure() 
    fig.update_layout(
        title='My title', 
        autosize=False,
        width=1000, 
        height=1000,
        margin=dict(l=65, r=50, b=65, t=90), 
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=1, y=1, z=z_ratio))
    
    if cross_section == True:
        sz = X.shape
        cross_section_x = X[int(np.floor(sz[0]/2)), :]
        cross_section_z = Zdata[int(np.floor(sz[0]/2)), :]

        # Plot the cross-section as a 2D line plot

        objs = [go.Scatter(
            x=cross_section_x,
            y=cross_section_z,
            mode='lines',
            name='Cross-section at y=0'
        )]

        fig.update_layout(
            title="Cross-section at y=0",
            xaxis_title="x",
            yaxis_title="z",
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=z_ratio)
        )
    fig = go.Figure(data = objs)
    fig.show()
    return

def main_signal():
    # set filename with its relative path
    path = r'C:\Users\egrab\Desktop\Unipi2024\RTDA\Lavoro_CSM2024\data_rtec\prove_acciaio_Flavio_39NiCrMo3\StribekR20_1minstep_20mps_100mmps_25C_VG46'
    filename = r'LOG.csv'
    path = r'C:\Users\egrab\Desktop\Unipi2024\RTDA\Closed_call_REOBTAIN\REOBTAIN\prova_26_05_2025_test_velocita_GrassoFerr'
    filename = r'LOG.csv'
    fullfilename = f"{path}\\{filename}"

    outputs, time, y_labels = read_signals(fullfilename, all_steps = True)
    plot_signal(outputs, time, y_labels, index='all', title = 'Stribeck', fontsize = 20)

    z_depth = outputs[3].copy()
    # z_depth = z_depth[50000:]
    # time = time[50000:]
    # z_depth = z_depth - z_depth.min()
    # d = np.sqrt(4.76**2 - (4.76 - z_depth)**2)

    # area = d
    # p = 10/area

    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.grid()
    # ax.set_ylabel('P (MPa)')
    # ax.set_xlabel('t (s)')
    # print(p.max())
    # # ax.plot(time, z_depth)
    # ax.plot(time, p)

    # show all plots
    plt.show()

def main_scan():
    # set filename with its relative path
    path = r'D:\disk_data\Rtec\TAMAVS\TAMVAS\Scansioni'
    filename = r'sede55041575_3124_Ix10_3D.bcrf'
    fullfilename = f"{path}\\{filename}"

    Zdata, info_data = read_bcrf(fullfilename)
    plot_3Dscan(Zdata, info_data, z_ratio = 0.3, crange = [400, 520], cross_section = False)


    # show all plots
    plt.show()

def main_test():

    import plotly.graph_objects as px
    import numpy as np
    # creating random data through randomint
    # function of numpy.random
    np.random.seed(42)
    
    # Data to be Plotted
    random_x = np.random.randint(1, 101, 100)
    random_y = np.random.randint(1, 101, 100)
    
    plot = px.Figure(data=[px.Scatter(
        x=random_x,
        y=random_y,
        mode='markers',)
    ])
    
    # Add dropdown
    plot.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["type", "scatter"],
                        label="Scatter Plot",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "bar"],
                        label="Bar Chart",
                        method="restyle"
                    )
                ]),
                direction="down",
            ),
        ]
    )
    
    plot.show()
if __name__ == '__main__':
    # main_scan()
    # main_test()
    main_signal()