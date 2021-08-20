from collections import Counter
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from matplotlib.gridspec import GridSpec

sns.set(palette='Set2')


def connections_map(df, save=False):
    flt_2017_sept = df[(df['flight_date'].dt.year == 2017) & (
        df['flight_date'].dt.month == 9)][['origin_airport', 'destination_airport']]
    edges = flt_2017_sept.values
    g = nx.from_edgelist(edges)

    sg = next(nx.connected_component_subgraphs(g))
    codes = np.unique(edges)

    geolocator = Nominatim(user_agent='geolocator')
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    df_codes = ['Airport ' + code for code in codes]
    df = pd.DataFrame(df_codes, columns=['code'])

    df['location'] = df['code'].apply(geocode).apply(
        lambda location: (location.latitude, location.longitude))

    deg = nx.degree(sg)
    sizes = [5 * deg[iata] for iata in sg.nodes]
    # labels = {iata: iata if deg[iata] >= 20 else ''
    #          for iata in sg.nodes}
    colors = [len(sg.edges(code)) for code in df['code'].values]
    df['code'] = df['code'].str.split(' ').str[1]
    pos = {key: value[::-1] for (key, value) in zip(df.code, df.location)}

    crs = ccrs.PlateCarree()
    _, ax = plt.subplots(
        1, 1, figsize=(12, 8),
        subplot_kw=dict(projection=crs))
    ax.coastlines()
    nx.draw_networkx(sg, ax=ax,
                     font_size=16,
                     alpha=.5,
                     width=.075,
                     node_size=sizes,
                     labels={},
                     pos=pos,
                     node_color=colors)
    if save:
        plt.savefig('distribution_map.png', bbox_inches='tight')


def cloud_coverage_dist(df, save=False):
    def cloud_cov_dist(df, coverages, axis):
        lvl1 = df[coverages[0]].value_counts() / df[coverages[0]].shape[0]
        lvl2 = df[coverages[1]].value_counts() / df[coverages[1]].shape[0]
        lvl3 = df[coverages[2]].value_counts() / df[coverages[2]].shape[0]

        data = pd.concat(
            [lvl1.rename('lvl1'), lvl2.rename('lvl2'), lvl3.rename('lvl3')], names=['lvl1', 'lvl2', 'lvl3'],
            axis=1, sort=True)

        data['index'] = data.index
        data = pd.melt(data, id_vars="index",
                       var_name="level", value_name="count")

        sns.catplot(x='index', y='count', hue='level',
                    data=data, kind='bar', ax=axis)
        axis.set_xlabel("")
        axis.set_ylabel("% of observations")
        plt.close(2)

    ori_cov = ['origin_cloud_coverage_lvl_1',
               'origin_cloud_coverage_lvl_2',
               'origin_cloud_coverage_lvl_3']

    dest_cov = ['destination_cloud_coverage_lvl_1',
                'destination_cloud_coverage_lvl_2',
                'destination_cloud_coverage_lvl_3']

    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    cloud_cov_dist(df, ori_cov, ax[0])
    cloud_cov_dist(df, dest_cov, ax[1])

    if save:
        plt.savefig('coverage_distribution.png', bbox_inches='tight')


def missing_values(df, save=False):
    all_data_na = (df.isnull().sum() / len(df)) * 100
    all_data_na = all_data_na.drop(
        all_data_na[all_data_na == 0].index).sort_values(ascending=False)

    plt.subplots(figsize=(10, 8))
    plt.xticks(rotation='90')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.ylabel('Percent of missing values', fontsize=15)
    if save:
        plt.savefig('missing_values.png', bbox_inches='tight', facecolor='w')


def correlation(df, algo='pearson', save=False):
    corrmat = df.corr(method=algo)
    plt.subplots()
    sns.heatmap(corrmat, vmax=0.9, square=True)
    if save:
        plt.savefig('correlation.png', bbox_inches='tight', facecolor='w')


def time_distribution(df, save=False):
    _, ax = plt.subplots(1, 3, figsize=(20, 5))

    year = df.groupby(df['flight_date'].dt.year).size()
    sns.barplot(x=year.index, y=year.values, ax=ax[0])
    ax[0].set_xlabel("")
    ax[0].set_title("Distribution of flights per year")

    month = df.groupby(df['flight_date'].dt.month).size()
    m = sns.barplot(month.index, month.values, ax=ax[1])
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUNE',
              'JULY', 'AUG', 'SEPT', 'OCT', 'NOV', 'DEC']
    m.set(xticklabels=months)
    ax[1].set_xlabel("")
    ax[1].set_title("Distribution of flights per month")

    week = df.groupby(df['flight_date'].dt.dayofweek).size()
    days_week = ['MON', 'TUES', 'WED', 'THURS', 'FRI', 'SAT', 'SUN']

    g = sns.barplot(week.index, week.values, ax=ax[2])
    g.set(xticklabels=days_week)
    ax[2].set_xlabel("")
    ax[2].set_title("Distribution of flights per week day")
    if save:
        plt.savefig('distribution_per_time.png', bbox_inches='tight')


def airc_model_fleet(df, save=False):
    with_fleet = df[df['fleet'].notna()]
    airc_model_perc = with_fleet.groupby(df['aircraft_model']).size().sort_values(
        ascending=False).head(10) / with_fleet['aircraft_model'].shape[0] * 100

    fleets = []
    for model in airc_model_perc.index:
        fleets.append(
            df.loc[(df['aircraft_model'] == model).idxmax(), 'fleet'])

    joint = pd.concat([airc_model_perc, pd.Series(
        fleets, index=airc_model_perc.index)], axis=1)
    joint = joint.rename(columns={0: 'percentage', 1: 'fleet'})

    _, ax = plt.subplots(figsize=(6, 4))
    g = sns.barplot(x=joint.index, y=joint['percentage'],
                    hue=joint['fleet'], data=joint, ax=ax, dodge=False)
    g.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("% of flights")
    ax.set_xlabel('')
    if save:
        plt.savefig('airc_model_fleet_dist.png', bbox_inches='tight')


def fleet_time_flt(df, save=False):
    to_plot = df.copy()
    to_plot['actual_block_time'] = round(
        (to_plot['on_block_date'] - to_plot['off_block_date']).dt.total_seconds()/60)
    short_haul = to_plot[(to_plot['actual_block_time'] >= 30) &
                         (to_plot['actual_block_time'] < 180)]
    medium_haul = to_plot[(to_plot['actual_block_time'] >= 180) &
                          (to_plot['actual_block_time'] < 360)]
    long_haul = to_plot[to_plot['actual_block_time'] >= 360]
    legend = []
    legend.append(short_haul['fleet'].value_counts().argmax())
    legend.append(medium_haul['fleet'].value_counts().argmax())
    legend.append(long_haul['fleet'].value_counts().argmax())
    idx = ['30min-3h', '3-6h', '> 6h']
    vals = []
    vals.append((short_haul[short_haul['fleet'] ==
                            legend[0]].shape[0] / short_haul.shape[0]) * 100)
    vals.append((medium_haul[medium_haul['fleet'] ==
                             legend[1]].shape[0] / medium_haul.shape[0]) * 100)
    vals.append((long_haul[long_haul['fleet'] == legend[2]
                           ].shape[0] / long_haul.shape[0]) * 100)

    joint = pd.concat([pd.Series(vals, index=idx),
                       pd.Series(legend, index=idx)], axis=1)
    joint = joint.rename(columns={0: 'percentage', 1: 'fleet'})

    _, ax = plt.subplots(figsize=(6, 4))
    g = sns.barplot(x=idx, y=joint['percentage'],
                    hue=joint['fleet'], data=joint, ax=ax, dodge=False)
    ax.set_ylabel("% of flights per category")
    ax.set_xlabel('')
    if save:
        plt.savefig('fleet_time_flt.png', bbox_inches='tight')


def tail_fleet(df, save=False):
    tail_number_perc = df.groupby(df['tail_number']).size().sort_values(
        ascending=False).head(100) / df['tail_number'].shape[0] * 100

    fleets = []
    for model in tail_number_perc.index:
        fleets.append(df.loc[(df['tail_number'] == model).idxmax(), 'fleet'])

    joint = pd.concat([tail_number_perc, pd.Series(
        fleets, index=tail_number_perc.index)], axis=1)
    joint = joint.rename(columns={0: 'percentage', 1: 'fleet'})

    _, ax = plt.subplots(figsize=(6, 4))
    g = sns.barplot(x=joint.index, y=joint['percentage'],
                    hue=joint['fleet'], data=joint, ax=ax, dodge=False)
    g.set_xticklabels("")
    ax.set_ylabel("% of flights")
    ax.set_xlabel('')
    if save:
        plt.savefig('tail_fleet.png', bbox_inches='tight')


def freq_connections(edges, save=False):
    list_edges = list(map(tuple, edges))
    counter = Counter(list_edges)
    top = counter.most_common(16)
    unzipped = list(zip(*top))

    labels = list(map('-'.join, list(unzipped[0])))
    values = np.array(list(unzipped[1])) / len(list_edges) * 100

    plt.subplots()
    plt.xticks(rotation='90')
    sns.barplot(x=labels, y=values)
    plt.ylabel('% of flights')
    if save:
        plt.savefig('frequent_connections.png', bbox_inches='tight')


def absolute_flt_pie(df, save=False):
    origin = df.groupby(
        'origin_airport').size().sort_values(ascending=False).head(8)
    destination = df.groupby(
        'destination_airport').size().sort_values(ascending=False).head(8)
    ori_sizes = origin.values
    dest_sizes = destination.values
    ori_labels = [s for s in origin.index]
    dest_labels = [s for s in destination.index]

    def airport_pie(ax, labels, sizes, title):
        explode = [0.3 if sizes[i] <
                   20000 else 0.0 for i in range(len(labels))]
        ax.pie(sizes, explode=explode, labels=labels, autopct=lambda p: '{:.0f}'.format(
            p * sum(sizes) / 100), startangle=90)
        ax.axis('equal')
        ax.set_title(title, fontsize=20)

    _, ax = plt.subplots(1, 2, figsize=(18, 10))
    airport_pie(ax[0], ori_labels, ori_sizes, 'Origin airports')
    airport_pie(ax[1], dest_labels, dest_sizes, 'Destination airports')
    if save:
        plt.savefig('airport_pies.png', bbox_inches='tight')


def delay_daily_pie(df, save=False):
    df2 = df[df['delay_minutes'].notna(
    )][['origin_airport', 'delay_minutes', 'scheduled_departure_date']]

    colors = sns.color_palette().as_hex()
    fig = plt.figure(1, figsize=(15, 9))
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    grouped1 = df2.groupby(df2['scheduled_departure_date'].dt.hour)[
        'delay_minutes'].count()
    labels1 = [s for s in grouped1.index]
    sizes1 = grouped1.values
    explode1 = [0.1 if sizes1[i] < 13000 else 0.0 for i in range(len(labels1))]
    ax1.pie(sizes1, explode=explode1,
            labels=labels1, colors=colors, autopct='%1.0f%%',
            shadow=False, startangle=0)
    ax1.axis('equal')
    ax1.set_title('% of delayed minutes per hour', fontsize=20)

    # ----------------------------------------
    grouped2 = df2.groupby(df2['scheduled_departure_date'].dt.hour)[
        'delay_minutes'].mean()
    labels2 = [s for s in grouped2.index]
    sizes2 = grouped2.values
    explode2 = [0.1 if sizes2[i] > 17.5 else 0.0 for i in range(len(labels2))]
    ax2.pie(sizes2, explode=explode2, labels=labels2,
            colors=colors, shadow=False, startangle=0,
            autopct=lambda p:  '{:.0f}'.format(p * sum(sizes2) / 100))
    ax2.axis('equal')
    ax2.set_title('Average minutes of delay per hour', fontsize=20)

    plt.tight_layout(w_pad=5)
    if save:
        plt.savefig('delay_hour.png', bbox_inches='tight')


def delay_sample(df, save=False):
    sch_dep = df[(df['flight_date'].dt.year == 2017) & (
        df['flight_date'].dt.month == 9) & (df['flight_date'].dt.day == 1)]
    sch_arr = sch_dep.sort_values(['scheduled_arrival_date'])

    _, ax = plt.subplots(2, 1, figsize=(14, 11))

    def plot_dates(ax, sch, act, label1, label2):
        ax.plot_date(sch, act, '-b', label=label1, linewidth=1.0)
        ax.plot_date(sch, sch, '--r', label=label2, linewidth=2.0)
        # ax.plot_date(sch, sch + timedelta(minutes=15), '--g',
        #             label='15min Delay Threshold', linewidth=2.0)
        # ax.grid(False)
        ax.set_ylabel(label1)
        ax.legend(loc='upper left')
        day = mdates.DayLocator()
        day_fmt = mdates.DateFormatter('%d')
        hour = mdates.HourLocator(interval=2)
        hour_fmt = mdates.DateFormatter('%H')
        ax.xaxis.set_major_locator(day)
        ax.xaxis.set_major_formatter(day_fmt)
        ax.xaxis.set_minor_locator(hour)
        ax.xaxis.set_minor_formatter(hour_fmt)
        ax.xaxis.set_tick_params(
            which='major', labelsize=14)
        ax.yaxis.set_major_locator(day)
        ax.yaxis.set_major_formatter(day_fmt)
        ax.yaxis.set_minor_locator(hour)
        ax.yaxis.set_minor_formatter(hour_fmt)
        ax.yaxis.set_tick_params(
            which='major', labelsize=14)

    plot_dates(ax[0], sch_dep['scheduled_departure_date'],
               sch_dep['off_block_date'], 'Off Block', 'Scheduled Departure')
    plot_dates(ax[1], sch_arr['scheduled_arrival_date'],
               sch_arr['on_block_date'], 'On Block', 'Scheduled Arrival')
    if save:
        plt.savefig('sch_act_plot.png', bbox_inches='tight')


def simple_bar(df, attr_name, save=False):
    perc = df[attr_name].value_counts().head(5) / df[attr_name].shape[0] * 100
    plt.subplots()
    sns.barplot(x=perc.index, y=perc.values)
    if save:
        plt.savefig(f'{attr_name}_dist.png', bbox_inches='tight')


def delay_month_weekday(df, save=False):
    _, ax = plt.subplots(figsize=(10, 8))
    delays = df[df['delay_code'].notna()]
    delays = delays.groupby([delays['flight_date'].dt.dayofweek, delays['flight_date'].dt.month])[
        'flight_date'].count().unstack().fillna(0)
    delays.T.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("# delayed flights")
    ax.set_xlabel("")
    days_week = ['MON', 'TUES', 'WED', 'THURS', 'FRI', 'SAT', 'SUN']
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUNE',
              'JULY', 'AUG', 'SEPT', 'OCT', 'NOV', 'DEC']
    plt.legend(days_week)
    ax.set_xticklabels(months)
    if save:
        plt.savefig('delay_distribution_month_weekday', bbox_inches='tight')


def proportion_delay_type(df, save=False):
    _, ax = plt.subplots(figsize=(10, 8))

    delayed = df[df['delay_code'].notnull()]
    df['delay_code'] = df['delay_code'].astype(int)
    delayed = df.groupby(['origin_airport', 'delay_code']).size().unstack(
        'delay_code')

    delayed1 = delayed.loc[delayed.sum(axis=1).sort_values(
        ascending=False).head(7).index]

    for index, row in delayed1.iterrows():
        delayed1.loc[index] = row.div(
            len(df[df['origin_airport'] == index]))

    delayed1 = delayed1.loc[:, delayed1.sum().sort_values(
        ascending=False).head(5).index]

    delayed1.T.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("proportion delayed flights")
    ax.set_xlabel("")
    #plt.title("Proportion of delayed flights per type and origin airport")
    if save:
        plt.savefig('proportion_delay_type_ori_airp', bbox_inches='tight')


def delay_calculated_source(df, with_text=False, save=False):
    df_copy = df.copy()
    df_copy['departure_delay'] = round(
        (df_copy['off_block_date'] - df_copy['scheduled_departure_date']).dt.total_seconds()/60)
    df_copy['departure_delay'] = df_copy['departure_delay'].astype(int)
    df_copy.loc[df_copy['departure_delay'] < 0, 'departure_delay'] = 0

    _, ax = plt.subplots(figsize=(10, 5))

    to_plot = df_copy.sample(frac=0.1)
    sns.scatterplot(x=to_plot['delay_minutes'],
                    y=to_plot['departure_delay'], ax=ax)
    #ax.set_title("Difference between given and calculated delays")
    ax.set_xlabel("Delay minutes from dataset")
    ax.set_ylabel("Calculated delay minutes")
    plt.xlim(0, 500)
    plt.ylim(0, 800)
    if with_text:
        joined = [df_copy['delay_minutes'], df_copy['departure_delay']]
        text = ""
        for i in range(2):
            if i == 0:
                text += "\n\nDataset delay"
            if i == 1:
                text += "\n\nCalculated delay"
                text += "\nMean: {}\nMax: {}\nMin: {}\n".format(
                        round(joined[i].mean()), joined[i].max(), joined[i].min())

        plt.text(-150, 150, text, fontsize=12)

    if save:
        plt.savefig('delay_differences.png', bbox_inches='tight')


def weather_distributions(df, save=False):
    def weather_dist(df, attr_names, pos, title):
        #ax = fig.add_subplot(3, 2, idx)
        ax = plt.subplot(pos)
        ori_attr = df[attr_names[0]][df[attr_names[0]].notna()]
        sns.distplot(ori_attr, label=attr_names[0], ax=ax)
        dest_attr = df[attr_names[1]][df[attr_names[1]].notna()]
        sns.distplot(dest_attr, label=attr_names[1], ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_yticklabels([])
        ax.legend()

    #_, ax = plt.subplots(3, 2, figsize=(12, 10))
    _ = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 4)

    weather_dist(df, ['origin_air_temperature',
                      'destination_air_temperature'], gs[0, 0:2], "Temperature")
    weather_dist(df, ['origin_wind_direction',
                      'destination_wind_direction'], gs[0, 2:], "Wind Direction")
    weather_dist(df, ['origin_wind_speed',
                      'destination_wind_speed'], gs[1, 0:2], "Wind Speed")
    weather_dist(df, ['origin_visibility',
                      'destination_visibility'], gs[1, 2:], "Visibility")
    weather_dist(df, ['origin_cloud_height',
                      'destination_cloud_height'], gs[2, 1:3], "Cloud Height")

    plt.tight_layout()
    if save:
        plt.savefig('weather_distributions.png', bbox_inches='tight')


def cloud_distribution(df, save=False):
    ori = df[(df['off_block_date'].dt.year == 2017) & (
        df['off_block_date'].dt.month == 9) & (df['origin_airport'] == 'LIS')]
    ori = ori[['off_block_date', 'origin_cloud_coverage', 'origin_cloud_height']]
    ori.rename(columns={'off_block_date': 'date',
                        'origin_cloud_coverage': 'coverage',
                        'origin_cloud_height': 'height'
                        }, inplace=True)

    dest = df[(df['on_block_date'].dt.year == 2017) & (
        df['on_block_date'].dt.month == 9) & (df['destination_airport'] == 'LIS')]
    dest = dest[['on_block_date', 'destination_cloud_coverage',
                 'destination_cloud_height']]
    dest.rename(columns={'on_block_date': 'date',
                         'destination_cloud_coverage': 'coverage',
                         'destination_cloud_height': 'height'
                         }, inplace=True)

    all = pd.concat([ori, dest])
    all = all.sort_values(by=['date'])

    _, ax = plt.subplots()

    first1 = all['date'].iloc[0] - timedelta(hours=1)
    last1 = all['date'].iloc[-1] + timedelta(hours=1)
    ax.set_xlim([first1, last1])
    ax.set_ylim([0, all['height'].max() + 500])
    ax = sns.scatterplot(x=all['date'], y=all['height'],
                         hue=all['coverage'], ax=ax, s=100)
    ax.set_xlabel('day')
    ax.set_ylabel('cloud height')
    ax.legend().texts[0].set_text("coverage")
    day = mdates.DayLocator(interval=2)
    day_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(day)
    ax.xaxis.set_major_formatter(day_fmt)

    plt.tight_layout(w_pad=5)
    if save:
        plt.savefig('cloud_distribution.png', bbox_inches='tight')
