#! /usr/bin/env python

"""
The locations of detectors are shown on QGIS. 

Author:         Guanqun Cao (guanqun.cao@volvocars.com)
Last Update:    Oct 5, 2020
"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pdb import set_trace
import datetime


class PemsVisual(): 
    def __init__(self, data_path):
        self._data_path     = data_path

    def _label_barplots(self, ax):
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2,
                p.get_height()*1.005),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

    def _label_barhplots(self, ax):
        for p in ax.patches:
                #percentage = '{:.1f}%'.format(p.get_width())
                x = p.get_x() + p.get_width() + .5
                y = p.get_y() + p.get_height()/2 - .2
                ax.annotate(str(p.get_width()), (x, y))

    def _get_fwy_length(self, fwy_file):
        df  = pd.read_csv(self._data_path + fwy_file, sep="\s+\t")
        df  = df[['Fwy', 'Total Miles']]
        return df

    def _get_distance_between_stations(self, metadata_file, fwy_file):
        import re
        df  = pd.read_csv(self._data_path + metadata_file, sep='\t')
        df  = df.astype({"Fwy": str, "Dir": str})
        fwy_ls = pd.read_csv(self._data_path + fwy_file).iloc[:,0].tolist()
        df['fwy_dir'] = df['Fwy'] + '-' + df['Dir']
        lookup_dict = {}
        for fwy in fwy_ls:
            lookup_dict[re.sub('[^0-9]','', fwy[:2]) + fwy[2:]] = fwy
        df = df.replace({'fwy_dir':lookup_dict})
        df = df[df['County'] == 37] 
        route_length = df.groupby('fwy_dir')['Abs_PM'].agg(np.ptp)
        return route_length
    
    def show_fwy_length(self, fwy_file):
        route_length = self._get_fwy_length(fwy_file)
        route_length['Total Miles'] = route_length['Total Miles'].round().astype(int)
        #route_length['Total Miles'] = route_length['Total Miles'].round(1).astype(int)
        #route_length['Total Miles'] = route_length['Total Miles'].round().astype(int)
        ax = route_length.plot.bar(x='Fwy', y='Total Miles', figsize = [21, 9], color = 'g')
        ax.set_xlabel("Freeway names")
        ax.set_ylabel("Freeway length (in Miles)")
        self._label_barplots(ax)
        plt.legend(loc='upper left')
        plt.title('The total miles of freeways in Bay Area.')
        plt.savefig("bay_fwy_length.jpg", dpi=150)
        plt.show()

    def show_delay_per_mile(self, bottleneck_file, fwy_file):
        df          = pd.read_csv(self._data_path + bottleneck_file, delimiter='\t')
        delay_df    = df.groupby('Fwy')['Avg Delay (veh-hrs)'].sum().sort_values(ascending=False).to_frame()
        delay_df['Fwy']  = delay_df.index
        delay_df = delay_df[delay_df['Fwy'] != 'I205-W']
        delay_df.index.name  = 'index' 
        route_length = self._get_fwy_length(fwy_file)
        delay_with_length_df = delay_df.merge(route_length, on='Fwy', how='left')
        delay_with_length_df['delay_per_mile'] = round(delay_with_length_df['Avg Delay (veh-hrs)'] / delay_with_length_df['Total Miles'], 1)
        delay_with_length_df = delay_with_length_df.sort_values(by='delay_per_mile', ascending=True)
        ax = delay_with_length_df.plot.barh(y='delay_per_mile', x='Fwy', figsize = [15, 9], color = 'g')
        self._label_barhplots(ax)
        plt.legend(loc='lower right')
        plt.title('Freeways by Delay Time per Mile (Speed<35mph) in Bay Area of Mar 2021.')
        ax.set_xlabel("Delay Time per Mile (veh-hrs/mile)")
        ax.set_ylabel("Freeway Names")
        plt.savefig("delay_per_mile_bay_03_2021.jpg", dpi=150)
        #plt.savefig("delay_per_mile_la_03_2021.jpg", dpi=150)
        plt.show()
        
    def show_bottleneck_distr(self, bottleneck_file):
        df  = pd.read_csv(self._data_path + bottleneck_file, sep='\t')
        bins1 = list(range(300))[0::50]
        bins2 = list(range(300,601))[0::100]
        bins = bins1 + bins2
        bins.append(3000)
        binned_data = pd.cut(df['Avg Delay (veh-hrs)'], bins=bins, include_lowest=True)
        fig, ax = plt.subplots()
        ax = binned_data.value_counts(sort=False).plot.bar(rot=0, color="g", figsize=(9,7))
        ax.set_xticklabels(['<50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-400', '400-500', '500-600', '>600'])
        ax.set_xlabel("Average Delay Time (vehicle-hours)")
        ax.set_ylabel("Number of Bottlenecks")
        self._label_barplots(ax) # put the values on top of each bar..
        #plt.title('Distribution of Bottlenecks in SF Bay Area 2019')
        plt.title('Distribution of Bottlenecks in LA County March 2021')
        plt.savefig("fwy_bottleneck_distr_03_2021_la.jpg", dpi=150)
        plt.show()

    def show_recent_bottleneck_distr(self, bottleneck_file):
        df  = pd.read_csv(self._data_path + bottleneck_file, sep='\t')
        bins1 = list(range(60))[0::5]
        bins2 = list(range(60, 241))[0::30]
        bins = bins1 + bins2
        bins.append(3000)
        binned_data = pd.cut(df['Avg Delay (veh-hrs)'], bins=bins, include_lowest=True)
        fig, ax = plt.subplots()
        ax = binned_data.value_counts(sort=False).plot.bar(rot=0, color="g", figsize=(15,6))
        ax.set_xticklabels(['<5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-90','90-120','120-150','150-180','180-210', '210-240', '>240'])
        ax.set_xlabel("Average Delay Time (vehicle-hours)")
        ax.set_ylabel("Number of Bottlenecks")
        self._label_barplots(ax) # put the values on top of each bar..
        #plt.title('Distribution of Bottlenecks During Afternoon in Sep-Oct 2020 in SF Bay Area')
        plt.title('Distribution of Bottlenecks in bay area of March 2021')
        #plt.title('Distribution of Bottlenecks in LA County of March 2021')
        plt.savefig("fwy_bottleneck_distr_03_2021_bay_recent.jpg", dpi=150)
        plt.show()

    def show_bottleneck_num(self, bottleneck_file):
        df  = pd.read_csv(self._data_path + bottleneck_file, sep='\t')
        occ_df      = df['Fwy'].value_counts()
        ax = occ_df.plot.bar(figsize = [15, 9], color = 'g')
        ax.set_xlabel("Freeway names")
        self._label_barplots(ax)
        plt.title('Freeways With Most Number of Bottlenecks.')
        ax.set_xlabel("Freeway Names")
        ax.set_ylabel("Number of Bottlenecks")
        plt.savefig("fwy_bottleneck_num.jpg", dpi=150)
        #plt.show()

    def show_bottleneck_delay_time(self, bottleneck_file):
        df  = pd.read_csv(self._data_path + bottleneck_file, sep='\t')
        delay_df    = df.groupby('Fwy')['Avg Delay (veh-hrs)'].sum().sort_values(ascending=True)
        delay_df    = delay_df.round().astype(int)
        ax = delay_df.plot.barh(figsize = [18, 10], color = 'g')
        self._label_barhplots(ax)
        #plt.title('Freeways by Delay Time (Speed<35mph) during morning in SF Bay of Sep-Oct 2020.')
        #plt.title('Freeways by Delay Time (Speed<35mph) in LA of Mar 2021.')
        plt.title('Freeways by Delay Time (Speed<35mph) in SF Bay of Mar 2021.')
        ax.set_xlabel("Aggregated Delay Time (veh-hrs)")
        ax.set_ylabel("Freeway Names")
        #plt.savefig("fwy_bottleneck_delay_0910_2020_am.jpg", dpi=150)
        plt.savefig("fwy_bottleneck_delay_03_2021_bay.jpg", dpi=150)
        #plt.savefig("fwy_bottleneck_delay_03_2021_la.jpg", dpi=150)
        plt.show()

    def compare_bottlenecks_2018_to_2020(self, file_prefix):
        years = range(2018, 2021)
        df_all = [] 
        for y in years:
            y = str(y)
            df  = pd.read_csv(self._data_path + file_prefix + '_' + y + '.tsv', sep='\t')
            delay_df    = df.groupby('Fwy')['Avg Delay (veh-hrs)'].sum().sort_values(ascending=False)
            delay_df    = delay_df.round().astype(int)
            df_all.append(pd.DataFrame({'Fwy_'+y:delay_df.index, 'delay_'+y:delay_df.values}))
        df_all = pd.concat(df_all, axis=1)
        df_all.to_csv(self._data_path + 'fwy_with_bottlenecks_2018_to_2020.csv', index=False)

if __name__ == '__main__':
    # Data directories 
    data_path = '/home/gcao/Datasets/cal_data/traffic/bay/'
    # File names
    #route_file = 'i10e_04022018.tsv'
    #metadata_file = 'd07_text_meta_2020_08_11.txt'
    #fwy_file = 'la_fwy.tsv'
    fwy_file = 'bay_fwy.tsv'
    bottleneck_file = 'bay_bottlenecks_03_2021.tsv'
    #bottleneck_file = 'la_county_bottlenecks_03_2021.tsv'
    file_prefix = 'bay_bottlenecks'
    pv = PemsVisual(data_path)
    #pv.show_fwy_length(fwy_file)
    # show bottlenecks 
    #pv.show_bottleneck_distr(bottleneck_file)
    #pv.show_recent_bottleneck_distr(bottleneck_file)
    #pv.show_bottleneck_delay_time(bottleneck_file)
    #pv.compare_bottlenecks_2018_to_2020(file_prefix)
    # calculate the delay per mile 
    pv.show_delay_per_mile(bottleneck_file, fwy_file)
