# %%
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from src import consts as const
from src.processing import attribute_builder as ab
from src.processing import plotting, refactor

sns.set(palette="Set2")


# Output Configurations
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 60)
plt.style.use('classic')

# Read Dataset
date_cols = ['flight_date', 'scheduled_departure_date', 'off_block_date', 'take_off_date',
             'landing_date', 'on_block_date', 'scheduled_arrival_date', 'registered_delay_date']

df = pd.read_csv(const.PROCESSED_DATA_DIR / 'full_info.csv',
                 sep='\t', parse_dates=date_cols)

# %% [markdown]
# ## Overview
df.head(5)

# %%
### PRELIMINARY SETUP ###
df.drop(',', axis=1, inplace=True)
df.rename(columns={'size_code': 'fleet'}, inplace=True)
print("The dataset size is: {}".format(df.shape))

# %%
types = df.dtypes
counts = df.apply(lambda x: x.count())
uniques = df.apply(lambda x: [x.unique()])
distincts = df.apply(lambda x: x.unique().shape[0])
missing_ratio = (df.isnull().sum() / df.shape[0]) * 100
cols = ['types', 'counts', 'uniques', 'distincts', 'missing_ratio']
desc = pd.concat([types, counts, uniques, distincts,
                  missing_ratio], axis=1, sort=False)
desc.columns = cols

# %%
### DELETE BASED ON FILLING FACTOR ###
df = refactor.remove_cols_nan_based(df, .7)  # remove cols with > 70% nan

# %%
delayed = df[df['delay_code'].notna()].shape[0]
not_delayed = df.shape[0] - delayed
plt.subplots()
sns.barplot(x=['delayed', 'not delayed'], y=[delayed, not_delayed])
plt.savefig('num_delayed.png', bbox_inches='tight')

# %%
# using only delayed flights
df.drop(df.loc[df['delay_code'].isna()].index, axis=0, inplace=True)

# %%
edges = df[['origin_airport', 'destination_airport']].values
g = nx.from_edgelist(edges)
print('There are {} different airports and {} connections'.format(
    len(g.nodes()), len(g.edges())))


# %%
plotting.connections_map(df)

# %%
plotting.freq_connections(edges, save=True)

# %%
plotting.absolute_flt_pie(df, save=True)

# %%
plotting.time_distribution(df, save=True)

# %%
plotting.simple_bar(df, 'fleet', save=True)

# %%
### ADJUST FLEET ###
df = refactor.adjust_fleets(df)

# %%
plotting.airc_model_fleet(df, save=True)

# %%
plotting.fleet_time_flt(df, save=True)

# %%
plotting.tail_fleet(df, save=True)

# %%
plotting.delay_daily_pie(df, save=True)

# %%
plotting.delay_sample(df, save=True)

# %%
### REMOVING BADLY FORMED RECORDS ###
same_ori_dest = df.loc[df['origin_airport'] ==
                       df['destination_airport']]
print(
    f"# of records with same origin and destination airports: {same_ori_dest.shape[0]}")
df.drop(same_ori_dest.index, axis=0, inplace=True)

not_take_off = df.loc[df['take_off_date'].isna()]
print(
    f"# of planes that did not take off after same ori-dest instances removed: {not_take_off.shape[0]}")
df.drop(not_take_off.index, axis=0, inplace=True)

not_landing = df.loc[df['landing_date'].isna()]
print(
    f"# of planes that did not land after same ori-dest instances removed: {not_landing.shape[0]}")
df.drop(not_landing.index, axis=0, inplace=True)

training_flt = df.loc[df['service_type'] == 'K']
print(f"# of training flights: {training_flt.shape[0]}")
df.drop(training_flt.index, axis=0, inplace=True)

nan_takeoff = len(df.loc[df['take_off_date'].isna()])
nan_landing = len(df.loc[df['landing_date'].isna()])
nan_offblock = len(df.loc[df['off_block_date'].isna()])
nan_onblock = len(df.loc[df['on_block_date'].isna()])
print(f"Null take-off: {nan_takeoff}")
print(f"Null landing: {nan_landing}")
print(f"Null off-block: {nan_offblock}")
print(f"Null on-block: {nan_onblock}")

offblock_takeoff = df.loc[df['off_block_date'] > df['take_off_date']]
print(f"off-block > take-off: {len(offblock_takeoff)}")
df.drop(offblock_takeoff.index, axis=0, inplace=True)

takeoff_landing = df.loc[df['take_off_date'] >= df['landing_date']]
print(f"take-off >= landing: {len(takeoff_landing)}")
df.drop(takeoff_landing.index, axis=0, inplace=True)

landing_onblock = df.loc[df['landing_date'] > df['on_block_date']]
print(f"landing > on-block: {len(landing_onblock)}")
df.drop(landing_onblock.index, axis=0, inplace=True)

print("\nThe dataset size is: {}".format(df.shape))

# %%
# plotting.delay_month_weekday(df)

# %%
plotting.proportion_delay_type(df, save=True)

# %%
# Build delay codes
df = refactor.build_delay_codes(df)

# %%
plotting.cloud_coverage_dist(df, save=True)

# %%
df = refactor.fix_cloud_data(df)
df = refactor.remove_cols_nan_based(df, .7)  # remove cols with > 70% nan

# %%
df.rename(
    columns={'origin_cloud_coverage_lvl_1': 'origin_cloud_coverage',
             'origin_cloud_height_lvl_1': 'origin_cloud_height',
             'destination_cloud_coverage_lvl_1': 'destination_cloud_coverage',
             'destination_cloud_height_lvl_1': 'destination_cloud_height'}, inplace=True)

# %%
plotting.weather_distributions(df, save=True)

# %%
plotting.cloud_distribution(df, save=True)


# %%
# Save data
df.to_csv(const.PROCESSED_DATA_DIR / 'basic_eda.csv',
          sep='\t', encoding='utf-8', index=False)


# %%
