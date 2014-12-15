import math
import os
import sys
import numpy as np
import matplotlib as mpl

if sys.platform == 'darwin':
    mpl.use('macosx')

import matplotlib.pyplot as plt
import matplotlib.widgets as mpl_w
import matplotlib.gridspec as gridspec
import csv
from datetime import datetime

### JBP imports to support email ...
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
import smtplib
import threading


import time
#import random

class EyeDataPlot:
    def __init__(self, filepath, coder, targetList=[], targetDuration=1.000, timeAfterTarget=0.125, fixationWindowSec = 0.250):
        self.readData(filepath)

        self.interests = ['time', \
                 'left_gaze_x','left_gaze_y', \
                 'right_gaze_x','right_gaze_y', \
                 'posx', 'posy', \
                 'ROW_INDEX']   # 0.702

        self.coder = coder  # Identifier for person doing coding
        self.targetDuration  = targetDuration  # sec
        self.timeAfterTarget = timeAfterTarget  # sec
        self.moveToNextTarget = False
        self.targetNumber = 0
        self.fixationWindowSec = fixationWindowSec
        self.filepath = filepath
        self.dirPath, self.fileName = os.path.split(filepath)
        self.csvFileName = filepath[:-4] + datetime.now().strftime('_%Y-%m-%d_%H-%M') + '.csv' # Output filename
        self.firstPass = True  # So we can initialize the plots
        #self.readyForClick = False

        if targetList == []:
            self.targetList = list(range(1,50))
            #random.shuffle(self.targetList)
        else:
            #removes targets that are not in the range [1,49]
            self.targetList = list(filter(lambda t:1<=t<=49, targetList))
            print(self.targetList)

        #global globalXYplotLimits
        global readyForClick

        self.XYplotLimits = [-30., 30., -100., 100.]  # Initialize to nonsense values ...
        readyForClick = False

        # Create output datafile, and write header
        self.csvfile = open(self.csvFileName, 'w')
        self.fileWriter = csv.writer(self.csvfile, delimiter=',')
        self.fileWriter.writerow(['Analysis file for: '+self.filepath+'  created: '+
                            datetime.now().strftime('%Y-%m-%d_%H-%M')])
        self.fileWriter.writerow(['Coded by', 'File', 'Eye', 'Target#', 'Start (sec)', 'End (sec)', 'Quality'])

    def readData(self, filepath):

        ###opening the file and reading in the data###
        file = open(filepath)
        lines = file.read().split('\n')
        self.table = []
        for line in lines:
            self.table.append( line.split('\t') )

    def extractData(self):
        ###preliminary processing###
        self.header = self.table.pop(0) #separate header from data

        i = 0
        while i < len(self.interests): #eliminate interests that are not in the data
            if self.header.count(self.interests[i]) == 0:
                self.interests.pop(i)
            else:
                i += 1

        self.indices = [ self.header.index(i) for i in self.interests ] #column indices of those interests
        
        ###data extraction###
        data = {} #all data
        nonan = {} #excludes "nan"s
        interests = self.interests

        for interest in interests: #keys are interests and values are intially empty lists
            data[interest] = []
            nonan[interest] = []

        for row in self.table:
            dats = [] #will contain selected values from this row

            if len(row) > 1: #excludes empty rows
                for k in self.indices: #add values in selected column to dats
                    dats.append( float(row[k]) )

                if [math.isnan(d) for d in dats].count(True) == 0: #to determine whether or not to include dats in nonan
                    addToNonan = True
                else:
                    addToNonan = False
                    
                for j in range(len(interests)): #append values from dats to their respective lists in data/nonan
                    data[ interests[j] ].append( dats[j] )

                    if addToNonan:
                        nonan[ interests[j] ].append( dats[j] )

        self.data = data
        self.nonan = nonan
        self.dataN = len(data['time']) #number of elements in data/nonan
        self.nonanN = len(nonan['time'])

        self.Hz = (self.dataN-1)/(data['time'][-1] - data['time'][0])   # calculate the hertz
        print('Averge sampling frequency = {:5.2f} Hz'.format(self.Hz)) # check on the speed

        self.filtMin = 1                                # median filter width min (1 sample),
        self.filtMax = int(self.Hz)                          # max (1 second), and
        self.filtVal = (self.filtMin+self.filtMax)//10  # default (1/10 of a second)

    def changeInterests(self, new_interests): #supposing there is interest in other data too
        self.interests = new_interests

    class figure: #not to be confused with plt.figure

        mpl.rcParams['toolbar'] = 'None'  # Disable toolbar on matplotlib windows

        class Cursor:
            def __init__(self, ax, timeWindow, showText=False, XYplotLimits=[] ):
                self.ax = ax
                self.ly = ax.axvline(color='k')    # the 1st vert line
                self.ly_2 = ax.axvline(color='k')  # the 2nd vert line
                self.lx_thresh = ax.axhline(color='r', linewidth=3)  # the horizontal quality threshold line
                self.timeWindow = timeWindow
                self.showText = showText
                self.mouseTimeVal = 0.0  # Initialize x-value of mouse position
                self.qualityMetric = 1.0  # Initialize y-value of mouse position

                self.XYplotLimits = XYplotLimits
                print(XYplotLimits)

                # text location in axes coords
                self.txt = ax.text( 0.22, 0.92, '', transform=ax.transAxes, color='r', size=20)
                
                self.lowQualityThreshold = XYplotLimits[2] - (XYplotLimits[3] - XYplotLimits[2])/20.0
                self.lx_thresh.set_ydata(self.lowQualityThreshold)

            def mouse_move(self, event):
                if not event.inaxes == self.ax: return  # Only continue if mouse is in one of the axes

                x, y = event.xdata, event.ydata
                self.mouseTimeVal = x

                if y < self.lowQualityThreshold:
                    self.txt.set_text( 'MARK AS LOW-QUALITY DATA')
                    self.qualityMetric = 0.0
                else:
                    self.txt.set_text( ' ')
                    self.qualityMetric = 1.0

                # update the line positions
                self.ly.set_xdata(x )
                self.ly_2.set_xdata(x + self.timeWindow )

                if self.showText:
                    self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )

                global readyForClick
                readyForClick = True

                #plt.pause(.01)

                plt.draw()
                #plt.pause(.01)

        def __init__(self, idnum, XYplotLimits):  ## init for Class figure

            self.fig = plt.figure(idnum, figsize=(16, 6), dpi=80) #the figure
            self.subs = {} #subplots
            self.axes = {} #axes for positioning widgets
            self.widgets = {} #sliders, buttons, etc.
            self.lines = {} #graphed lines

            self.XYplotLimits = XYplotLimits

            #  subplots
            gs = gridspec.GridSpec(12, 2)  # 12 rows, 2 columns
            self.subs['time_xy_sub'] = self.fig.add_subplot(gs[0:8,0]) #top left, has raw data, filtered data, and target positions
            self.subs['error_sub'] = self.fig.add_subplot(gs[9:12,0], sharex=self.subs['time_xy_sub']) #bottom left, has Pythagorean error
            self.subs['velocity_sub'] = self.subs['error_sub'].twinx() #and velocity
            self.subs['x_vs_y_sub'] = self.fig.add_subplot(gs[0:8,1]) #right, has eye trace and target grid
            gs.update(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.10, hspace=0.05)

            #widgets
            self.axes['median'] = plt.axes([0.6,0.2, 0.35,0.02])
            self.axes['target'] = plt.axes([0.6,0.1, 0.35,0.02])
            
        ###functions for plotting stuff###

        def calcXYGrid(self, data, attrs): #identifies distinct target points

            #globalXYplotLimits = self.XYplotLimits

            gridXpt = [data['posx'][0]] # start with the first target point
            gridYpt = [data['posy'][0]]
            dataN = len(data['time'])
            
            for k in range(dataN):
                if data['posx'][k] != gridXpt[-1] or data['posy'][k] != gridYpt[-1]: #new target point location that's different from the most recent
                    gridXpt.append(data['posx'][k]) #add new target point location to list
                    gridYpt.append(data['posy'][k])

            self.XYplotLimits = [min(gridXpt)*1.5, max(gridXpt)*1.5, min(gridYpt)*1.5, max(gridYpt)*1.5]

            return (gridXpt,gridYpt) #return target x and y together

        def graphXYGrid(self, data, attrs, sub, style='x'): #2D graph for target grid
            (calGridX, calGridY) = self.calcXYGrid(data, attrs) #get target grid xs and ys and graph them

            self.plotXvsY(data, attrs, sub, style)

            sub.set_xlim([self.XYplotLimits[0] - 5, self.XYplotLimits[1] + 5]) #don't autozoom out if there are data points far away from target grid
            sub.set_ylim([self.XYplotLimits[2] - 3, self.XYplotLimits[3] + 3])

        def plotDataVsTime(self, data, attrs, sub, style='-', mfc=""): #graph data lists that correspond to the ones in attrs
            for attr in attrs[1:]: #for each attribute, plot that data against time
                if mfc != "":
                    line, = sub.plot(data[attrs[0]], data[attr], style, label=attr, markerfacecolor=mfc)
                else:
                    line, = sub.plot(data[attrs[0]], data[attr], style, label=attr)
                self.lines[attr] = line

            if attr == 'velocity':
                sub.legend(bbox_to_anchor=(0., 0., 1., 1.03), loc='upper left', prop={'size':10}) #creates legend and makes text smaller so the box doesn't take up too much space
            elif attr == 'pyth_err':
                sub.legend(bbox_to_anchor=(0., 0., 1., 1.04), loc='upper right', prop={'size':10}) #creates legend and makes text smaller so the box doesn't take up too much space
            else:  # X & Y, raw & filtered, posx & posy
                sub.legend(bbox_to_anchor=(0., 0., 1., 1.08), loc='upper left', ncol=1, prop={'size':8}) #creates legend and makes text smaller so the box doesn't take up too much space

            sub.set_ylim([self.XYplotLimits[2] - 3, self.XYplotLimits[3] + 3])

        def plotXvsY(self, data, attrs, sub, style='-',mfc=''): #2D graph plotting attrs[0] versus attrs[1]
            if attrs[0].count('_') > 0:
                side = attrs[0][: attrs[0].index('_') ] #will be 'left' or 'right'
            else:
                side = attrs[0] #generally, 'posx'

            if mfc != "":
                line, = sub.plot(data[attrs[0]], data[attrs[1]], style, label=side, markerfacecolor=mfc) #plots the data
            else:
                line, = sub.plot(data[attrs[0]], data[attrs[1]], style, label=side) #plots the data

            self.lines[side] = line

            sub.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=7, ncol=2, prop={'size':10}) #creates the legend
            return line #useful for getting the eye trace

        def addSlider(self, name, axName, sMin, sMax, sVal): #for median filter width and target sliders  0.702 threshold -> target

            self.widgets[axName+"_slider"] = mpl_w.Slider(self.axes[axName], name, sMin,sMax,sVal)
            return self.widgets[axName+"_slider"]

        def addCursor(self, axName, timeWindow, showText=False, ): #  0.708

            ax = self.subs[axName]

            cursor = self.Cursor(ax, timeWindow, showText=False, XYplotLimits = self.XYplotLimits)
            self.widgets[axName+'_cursor'] = cursor

            return self.widgets[axName+"_cursor"]

        def getXYplotLimits(self):
            return self.XYplotLimits

    def makeFigs(self): #automatically generates figure(s) for left and/or right eye(s)
        self.extractData() #extract relevant data from all data

        ###plot stuff###
        def createFigure(attrs, iden):

            global globalVmax

            fig = self.figure(iden, XYplotLimits=self.XYplotLimits) #create a pyplot figure

            x = attrs[0] #'left_gaze_x' or 'right_gaze_x'
            y = attrs[1] #'left_gaze_y' or 'right_gaze_y'

            data = self.data
            nonan = self.nonan

            filts = self.medfilt(data, ['time',x,y], self.filtVal) #filter time, x, and y
            nonan.update(filts) #because nans were excluded

            #2D plot
            xy_sub = fig.subs['x_vs_y_sub']
            
            fig.lines['trace'] = fig.plotXvsY(nonan, [x,y], xy_sub, style='o-', mfc='none')  # 2D eye trace
            fig.graphXYGrid(data, ['posx','posy'], xy_sub) #target grid

            self.XYplotLimits = fig.getXYplotLimits()

            #data vs time
            self.t_xy_sub = fig.subs['time_xy_sub']

            fig.plotDataVsTime(data, ['time',x,y], fig.subs['time_xy_sub'], style='.-') #raw data
            fig.plotDataVsTime(nonan, ['time_filt',x+'_filt',y+'_filt'], fig.subs['time_xy_sub'], style='o-', mfc='none') #filtered data
            fig.plotDataVsTime(data, ['time','posx','posy'], fig.subs['time_xy_sub'], style='--') #target position

            fig.addCursor('time_xy_sub', self.fixationWindowSec)  # Add special cursor to select time window in the top-left plot

            self.t_xy_sub.set_ylim(min(self.XYplotLimits[0], self.XYplotLimits[2]),
                                   max(self.XYplotLimits[1], self.XYplotLimits[3]))

            #error/velocity sub plots
            P = calculatePythagoreanError(x,y, data, nonan)
            nonan['pyth_err'] = P #again, because nans are excluded
            err_sub = fig.subs['error_sub'] #get only the subplot for Pythagorean error
            fig.plotDataVsTime(nonan, ['time_filt','pyth_err'], err_sub, style='b.-') #graph Pythagorean error by time

            err_sub.set_ylim([0, sum(P)/len(P)]) #set upper limit to mean of Pythagorean error

            V = calculateUndirectedVelocity(x,y, data, nonan)
            nonan['velocity'] = V #again, because nans are excluded
            globalVmax = 10.*sum(V)/len(V)
            nonan['time_filt_v'] = nonan['time_filt'][1:] #there is one fewer data point in velocity
            vel_sub = fig.subs['velocity_sub'] #get only the subplot for velocity
            vel_sub.set_ylim([0, globalVmax])  #set upper limit to mean of velocity

            fig.plotDataVsTime(nonan, ['time_filt_v','velocity'], vel_sub, style='g-') #graph velocity

            ###add sliders###
            #median filter
            fig.addSlider("Median\nfilter\nwidth", "median", self.filtMin,self.filtMax,self.filtVal)

            # threshold -> target 0.702
            targetSlider = fig.addSlider("Target", "target", 0, 48, self.targetNumber)  # , valfmt='%3.2f'

            return fig

        def calculatePythagoreanError(x,y, data, nonan): #Pythagorean error
            P = []
            pos = 0
            
            for k in range(len(nonan['time_filt'])):
                eye_x = nonan[x+'_filt'][k]
                eye_y = nonan[y+'_filt'][k] #eye position

                while data['time'][pos] < nonan['time_filt'][k]: #used to match data position with nonan position by time
                    pos += 1
                targ_x = data['posx'][pos]
                targ_y = data['posy'][pos] #target position
                
                err = math.sqrt( (eye_x - targ_x)**2 + \
                                 (eye_y - targ_y)**2) #Pythagorean distance between eye position and target position
                P.append(err)

            return P

        def calculateUndirectedVelocity(x,y, data, nonan): #calculate undirected velocity
            V = []
            for k in range(len(nonan['time_filt'])-1):
                eye_x1 = nonan[x+'_filt'][k]
                eye_y1 = nonan[y+'_filt'][k]
                eye_x2 = nonan[x+'_filt'][k+1] #successive eye positions
                eye_y2 = nonan[y+'_filt'][k+1]

                distance = math.sqrt( (eye_x2-eye_x1)**2 + \
                                      (eye_y2-eye_y1)**2 ) #Pythagorean distance between successive eye positions
                timediff = nonan['time_filt'][k+1] - nonan['time_filt'][k] #delta time
                velocity = distance / timediff #v = d/t
                V.append(velocity)
                
            return V

        startTime = self.data['time'][0] #initialize time range to earliest and latest times
        endTime = self.data['time'][-1]

        figs = [] #create and keep figure(s)
        if 'left_gaze_x' in self.interests:
            figs.append( [1, startTime, endTime, 'left_gaze', 0, \
                          createFigure(['left_gaze_x','left_gaze_y'], 1)] )
        if 'right_gaze_x' in self.interests:
            figs.append( [2, startTime, endTime, 'right_gaze', 0, \
                          createFigure(['right_gaze_x','right_gaze_y'], 2)] )

        ### structure of figs
        # figs
        #   left fig
        #     1
        #     <startTime>
        #     <endTime>
        #     "left_gaze"
        #     <initial target number>
        #     figure
        #   right fig
        #     ...
        ### So the pyplot figure for the left eye is figs[0][-1][0] (or for the right eye if there is no left eye)

        def fetchDataByTime(data, attrs, startTime, endTime): #gets the data in a particular time range
            beg = 0
            end = len(data['time'])

            #finds beginning position of startTime
            while beg < len(data['time'])-1 and data['time'][beg+1] < startTime:
                beg += 1

            #finds ending position of endTime
            while end > 0 and data['time'][end-1] > endTime:
                end -= 1

            return (data[attrs[0]][beg:end+1], data[attrs[1]][beg:end+1]) #x and y in time range

        def fetchDataByRowIndex(data, attrs, RowIndex): # gets the data for a particular target point

            #finds beginning position of RowIndex in ROW_INDEX
            first = data['ROW_INDEX'].index(RowIndex)
            startTime = data['time'][first]

            endTime   = startTime + self.targetDuration + self.timeAfterTarget  # go to end of target + 250 msec

            #find ending position of endTime
            last = len(data['time'])
            # Step through from end until you get to there
            while last > 0 and data['time'][last-1] > endTime:
                last -= 1

            return data[attrs[0]][first:last+1], data[attrs[1]][first:last+1], startTime, endTime #x and y after target point


        for figure in figs:
            t_xy_sub = figure[-1].subs['time_xy_sub'] #retrieve top left subplot
            (sT, eT) = t_xy_sub.get_xlim() #retrieve time range
            
            #self.targetNumber = 0#self.targetList[0]  # JBP #### LTB

            # We want to select the data with ROW_INDEX == nextTarget (and perhaps another 250msec)
            (xdats, ydats, sT, eT) = fetchDataByRowIndex(self.nonan, [figure[3]+'_x',figure[3]+'_y'],
                                                         self.targetList[figure[4]]) #re-get the data

            trace = figure[-1].lines['trace']
            trace.set_xdata(xdats) #update x and y data of eye trace
            trace.set_ydata(ydats)

            t_xy_sub.set_xlim(sT, eT)
            #t_xy_sub.set_ylim(globalXYplotLimits[2], globalXYplotLimits[3])
            t_xy_sub.set_ylim(min(self.XYplotLimits[0], self.XYplotLimits[2]),
                              max(self.XYplotLimits[1], self.XYplotLimits[3]))
            t_xy_sub.set_title('Target #' +
                               str(self.targetList[figure[4]]))

            figure[1] = sT #new time range is now in effect
            figure[2] = eT

            #figure[-1].fig.canvas.draw() #redraw the figure
        #plt.pause(.1)
        plt.draw()


        def updateDisplayByTarget(event):

            global readyForClick

            for figure in figs:
                if figure[-1].fig.canvas == event.canvas and readyForClick: #meaning I clicked in this figure
                    readyForClick = False

                    #update eye trace, if needed
                    t_xy_sub = figure[-1].subs['time_xy_sub'] #retrieve top left subplot
                    (sT, eT) = t_xy_sub.get_xlim() #retrieve time range

                    cursorPosition = figure[-1].widgets['time_xy_sub_cursor'].mouseTimeVal
                    qualityMetric = figure[-1].widgets['time_xy_sub_cursor'].qualityMetric

                    vel_sub = figure[-1].subs['velocity_sub'] #retrieve velocity subplot
                    vel_sub.set_ylim(0,globalVmax)

                    if figure[-1].subs["time_xy_sub"] == event.inaxes:  # If the mouse_click was in the upper-left subplot

                        self.fileWriter.writerow([self.coder,                                # Coder ID ('anon' default)
                                             self.fileName,                             # File data read from
                                             figure[3],                                      # 'right_' or 'left_gaze'
                                             self.targetList[figure[4]],    # Which target?
                                             cursorPosition,                            # Window start
                                             cursorPosition + self.fixationWindowSec,   # Window end
                                             qualityMetric])                            # High or Low quality
                        self.csvfile.flush()
                        
                        # move on to next target
                        figure[4] += 1

                        if figure[4] > len(self.targetList)-1: # End of trial...
                            #I would prefer to find some way to clear all figures, but this will work for now
                            plt.close(figure[0]) #close figure if all targets have been looked at

                            if len(figs) == 0:
                                raise SystemExit #exit program when all figures are closed
                            else:
                                return

                        targetSlider = figure[-1].widgets['target_slider']
                        targetSlider.set_val(self.targetList[figure[4]])

                        (xdats, ydats, sT, eT) = fetchDataByRowIndex(self.nonan, [figure[3]+'_x',figure[3]+'_y'],
                                                                     self.targetList[figure[4]]) #re-get the data

                        trace = figure[-1].lines['trace']
                        trace.set_xdata(xdats) #update x and y data of eye trace
                        trace.set_ydata(ydats)

                        t_xy_sub.set_xlim(sT, eT)
                        t_xy_sub.set_title('Target #' +
                                           str(self.targetList[figure[4]]))
                        # print(sT, eT)

                        figure[1] = sT #new time range is now in effect
                        figure[2] = eT

                        t_xy_sub.set_ylim(min(self.XYplotLimits[0], self.XYplotLimits[2]),
                                          max(self.XYplotLimits[1], self.XYplotLimits[3]))

                        #figure[-1].fig.canvas.draw() #redraw the figure
                        #plt.pause(.1)
                        plt.draw()

                    if sT != figure[1] or eT != figure[2]: #if the time range changed...
                        (xdats, ydats) = fetchDataByTime(self.nonan, [figure[3]+'_x',figure[3]+'_y'], sT, eT) #re-get the data
                        trace = figure[-1].lines['trace']
                        trace.set_xdata(xdats) #update x and y data of eye trace
                        trace.set_ydata(ydats)

                        figure[1] = sT #new time range is now in effect
                        figure[2] = eT

                        t_xy_sub.set_ylim(min(self.XYplotLimits[0], self.XYplotLimits[2]),
                                          max(self.XYplotLimits[1], self.XYplotLimits[3]))

                        #plt.pause(.1)

                        figure[-1].fig.canvas.draw() #redraw the figure

                    #update filtered data and lines
                    filtSlider = figure[-1].widgets['median_slider']
                    filtSlider.set_val( round(filtSlider.val) )
                    if filtSlider.val != self.filtVal:
                        self.filtVal = int(filtSlider.val)
                        refilterData(figure[-1], ['time',figure[3]+'_x',figure[3]+'_y'])

                    #make target slider value a whole number
                    targetSlider = figure[-1].widgets['target_slider']
                    targetSlider.set_val( round(targetSlider.val) )

##                    if targetSlider.val != self.targetNumber:
##                        #self.targetNumber = int(targetSlider.val)
##                        self.targetNumber = self.targetList.index(int(targetSlider.val))
##
##                        # We want to select the data with ROW_INDEX == nextTarget (and perhaps another 250msec)
##                        nextTarget = self.targetNumber
##                        #self.targetNumber = nextTarget
##
##                        (xdats, ydats, sT, eT) = fetchDataByRowIndex(self.nonan, [figure[3]+'_x',figure[3]+'_y'],
##                                                                     self.targetNumber) #re-get the data
##
##                        trace = figure[-1].lines['trace']
##                        trace.set_xdata(xdats) #update x and y data of eye trace
##                        trace.set_ydata(ydats)
##
##                        t_xy_sub.set_xlim(sT, eT)
##                        # print(sT, eT)
##
##                        figure[1] = sT #new time range is now in effect
##                        figure[2] = eT
##
##                        t_xy_sub.set_title('Target #' + str(nextTarget))

        def closeFigure(event): #remove figure from figs
            for figure in figs:
                if not plt.fignum_exists(figure[0]): #if figure id number doesn't exist, figure doesn't exist
                    print("removing figure",figure[0])
                    figs.remove(figure) #remove from figure list

            if len(figs) == 0:
                raise SystemExit #exit program if all figures have been closed

        for figure in figs: #for each figure connect events
            figure[-1].fig.canvas.mpl_connect('button_release_event', updateDisplayByTarget) #when there's a click, update trace if needed
            figure[-1].fig.canvas.mpl_connect('close_event', closeFigure) #if the 'x' button is clicked, remove figure from figs list

            figure[-1].fig.canvas.mpl_connect('motion_notify_event', figure[-1].widgets['time_xy_sub_cursor'].mouse_move)


        def refilterData(fig, toRefilter):

            nonan = self.nonan

            filts = self.medfilt(self.data, toRefilter, self.filtVal) #filter time, x, and y
            nonan.update(filts)
            
            #update dependent graph lines
            xVtime = fig.lines[toRefilter[1]+'_filt'] #filtered x vs time
            xVtime.set_xdata(nonan[toRefilter[0]+'_filt'])
            xVtime.set_ydata(nonan[toRefilter[1]+'_filt'])

            yVtime = fig.lines[toRefilter[2]+'_filt'] #filtered y vs time
            yVtime.set_xdata(nonan[toRefilter[0]+'_filt'])
            yVtime.set_ydata(nonan[toRefilter[2]+'_filt'])

            #Pythagorean error
            P = calculatePythagoreanError(toRefilter[1],toRefilter[2], self.data, nonan)
            nonan['pyth_err'] = P #again, because nans are excluded
            pythErr = fig.lines['pyth_err']
            pythErr.set_xdata( nonan['time_filt'] )
            pythErr.set_ydata( nonan['pyth_err'] )

            #undirected velocity
            V = calculateUndirectedVelocity(toRefilter[1],toRefilter[2], self.data, nonan)
            nonan['velocity'] = V #again, because nans are excluded
            nonan['time_filt_v'] = nonan['time_filt'][1:] #there is one fewer data point in velocity
            velLine = fig.lines['velocity']
            velLine.set_xdata( nonan['time_filt_v'] )
            velLine.set_ydata( nonan['velocity'] )

            #fig.fig.canvas
            ##plt.pause(.1)
            plt.draw()

        plt.show()

    def medfilt(self, data, attrs, width): #filters value lists corresponding to keys in attrs

        F = {}
        for a in attrs:
            F[a+"_filt"] = [] #initialize value lists to empty lists for each key in attrs

        if width > 0: #width is positive, so median is well-defined
            
            off = width//2 #offset from center for before/after bounds

            for q in range(off, len(data['time'])-off): #range of width "width" centered on current position
                for x, a in enumerate(attrs):
                    chunk = data[a][q-off : q+off+1] #data chunk centered on current position
                    scrub = [] #will have no nans
                    for c in chunk:
                        if not math.isnan(c):
                            scrub.append(c) #if c is not nan, append to scrub

                    if len(scrub) > off: #fewer than half were nans
                        F[a+"_filt"].append( np.median( scrub ) ) #calculate median on what's left
                    else: #at least half were nans, meaning the median is nan/undefined
                        #erase already-inputted values
                        for y in range(x):
                            a = attrs[y]
                            F[a+"_filt"].pop()
                        break #don't continue through attrs keys; waste of time/CPU

        return F
    
def email_data(dataFileName):  ### JBP email data file
    fromaddr = 'emra.data@gmail.com'
    toaddrs = 'emra.data@gmail.com'

    username = 'emra.data@gmail.com'
    password = 'saccade1!'

    print("sending {} to gmail ...".format(dataFileName))

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()

    msg = MIMEMultipart()

    frTo = "\r\n".join([
      "From: "+fromaddr,
      "To: "+toaddrs,
      "Subj: Datafile"
      ])

    msg.attach(frTo)

    f = file(dataFileName)
    attachment = MIMEText(f.read())
    attachment.add_header('Content-Disposition', 'attachment', filename=dataFileName)

    msg.attach(attachment)


    # attachments.attach(MIMEText(msg))

    # attachments.attach(MIMEImage(file(photoFileName).read()))   ### This is how you attach an image ...


    # attachments.attach('This is, the first line')
    # attachments.attach('This is the second, line')

    # csvReader = csv.reader(dataFileName)
    # for row in csvReader:
    #     attachments.attach(MIMEText(row))

    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, attachment.as_string())
    server.quit()

    print("                                     {} SENT to gmail.".format(dataFileName))
    # ======================================================================



def run(name, coder='anon', targets=[]):
    #random.seed(0)
    
    myEyeDataPlot = EyeDataPlot(name, coder, targets,
                                targetDuration=1.000,
                                timeAfterTarget=0,
                                fixationWindowSec = 0.100) #reads data

    try:
        myEyeDataPlot.makeFigs() #makes the figures

    except SystemExit: #normal execution exit
        return
    #except KeyboardInterrupt: #Ctrl+C
    #    raise KeyboardInterrupt
    except: #all other exceptions
        e = sys.exc_info()[0]
        print( "Error: %s" % e )
        raise e

def run2(filepath, coder='anon'):

    if sys.version_info[0] == 2:
        strfunc = unicode
    else:
        strfunc = str

    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        #lines = f.read().split('\r\n')
        coder, folderpath = lines[0].split(',')
        print(coder,'\t',folderpath)

        for L in lines[1:]:
            Lsplit = L.split(',')
            filename = Lsplit[0]
            print(Lsplit)

            if Lsplit[1] == 'r': #random mode
                if len(Lsplit) > 2 and strfunc(Lsplit[2]).isnumeric():
                    n = int(Lsplit[2])

                    if len(Lsplit) > 3 and strfunc(Lsplit[3]).isnumeric():
                        seed = int(Lsplit[3])
                    else:
                        seed = 0
                        
                else:
                    n = 49

                import random
                random.seed(seed)

                targets = list(range(1,49))
                random.shuffle(targets)
                targets = targets[:n]

            else:
                targets = [int(t) for t in Lsplit[1:]]
                
            print(' ',filename,'\t',targets)
            run(folderpath+filename, coder, targets)

if __name__ == '__main__':
    import os.path
    #if not os.path.isfile("codedFiles.txt"):

#        if sys.version_info[0] == 2 and sys.platform == 'windows':
#            inputFunc = raw_input
#        else:
#            inputFunc = input

#        coderName = inputFunc("Please enter your initials: ")

#        if sys.platform == 'darwin':
#            coderName = str(coderName)
            
#        with open("codedFiles.txt", 'w') as file:
#            file.write(coderName+'\n')

    ### JBP email a datafile
    fname = 'test.csv'
    emailThread = threading.Thread(target=email_data, args=([fname]))
    emailThread.start()

    time.sleep(10)





    with open("./codedFiles.txt", 'r') as doneFileList:
        #print(doneFileList.read())
        lines = doneFileList.read().splitlines()
        print(lines)

        if len(lines) > 1:
            coder, doneFiles = lines[0], lines[1:]
            print(doneFiles)
        else:
            coder = lines[0]
            doneFiles = []

    with open("dataFileList.txt", 'r') as dataFileList:
        files = dataFileList.read().splitlines()
        for line in files:
            if line.split(',')[0] not in doneFiles:
                print(line)
                break

        if line != files[-1] or line.split(',')[0] not in doneFiles:
            
            with open("inputFile.txt", 'w') as inputFile:
                inputFile.write(coder+',Data files/\n')
                inputFile.write(line+'\n')

            with open("codedFiles.txt", 'a') as doneFileList:
                doneFileList.write(line.split(',')[0]+'\n')

            run2("inputFile.txt")
