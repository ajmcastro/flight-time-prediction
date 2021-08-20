

def departure_delay_creation(df):
    df['departure_delay'] = round(
        (df['off_block_date'] - df['scheduled_departure_date']).dt.total_seconds()/60)
    df['departure_delay'] = df['departure_delay'].astype(int)
    df.loc[df['departure_delay'] < 0, 'departure_delay'] = 0
    return df


def arrival_delay_creation(df):
    df['arrival_delay'] = round(
        (df['on_block_date'] - df['scheduled_arrival_date']).dt.total_seconds()/60)
    df['arrival_delay'] = df['arrival_delay'].astype(int)
    df.loc[df['arrival_delay'] < 0, 'arrival_delay'] = 0
    return df


def air_time_creation(df):
    landing_takeoff = df[df['landing_date'] <= df['take_off_date']]
    print(f"Landing <= TakeOff: {landing_takeoff.shape[0]}")
    df.drop(landing_takeoff.index,
            axis=0, inplace=True)
    df['air_time'] = round(
        (df['landing_date'] - df['take_off_date']).dt.total_seconds()/60)
    df['air_time'] = df['air_time'].astype(int)
    return df


def taxi_out_creation(df):
    takeoff_offblock = df[df['take_off_date'] < df['off_block_date']]
    print(f"TakeOff < OffBlock: {takeoff_offblock.shape[0]}")
    df.drop(takeoff_offblock.index,
            axis=0, inplace=True)
    df['taxi_out'] = round(
        (df['take_off_date'] - df['off_block_date']).dt.total_seconds()/60)
    df['taxi_out'] = df['taxi_out'].astype(int)
    return df


def taxi_in_creation(df):
    onblock_landing = df[df['on_block_date'] < df['landing_date']]
    print(f"OnBlock < Landing: {onblock_landing.shape[0]}")
    df.drop(onblock_landing.index,
            axis=0, inplace=True)
    df['taxi_in'] = round(
        (df['on_block_date'] - df['landing_date']).dt.total_seconds()/60)
    df['taxi_in'] = df['taxi_in'].astype(int)
    return df


def scheduled_block_time_creation(df):
    arr_dep = df[df['scheduled_arrival_date'] <=
                 df['scheduled_departure_date']]
    print(f"Arrival <= Departure: {arr_dep.shape[0]}")
    df.drop(arr_dep.index, axis=0, inplace=True)
    df['scheduled_block_time'] = round(
        (df['scheduled_arrival_date'] - df['scheduled_departure_date']).dt.total_seconds()/60)
    df['scheduled_block_time'] = df['scheduled_block_time'].astype(int)
    return df


def actual_block_time_creation(df):
    on_off_block = df[df['on_block_date'] <= df['off_block_date']]
    print(f"OnBlock <= OffBlock: {on_off_block.shape[0]}")
    df.drop(on_off_block.index,
            axis=0, inplace=True)
    df['actual_block_time'] = round(
        (df['on_block_date'] - df['off_block_date']).dt.total_seconds()/60)
    df['actual_block_time'] = df['actual_block_time'].astype(int)
    return df


def create_targets(df):
    df = air_time_creation(df)
    df = taxi_out_creation(df)
    df = taxi_in_creation(df)
    df = actual_block_time_creation(df)
    return df


def is_night_creation(df):
    df['is_night'] = (df['scheduled_departure_date'].dt.hour >= 21) | (
        df['scheduled_departure_date'].dt.hour <= 3)
    return df


def create_scheduled_rotation_time(df):
    def compute_rotation_time(group):
        group.sort_values(by=['scheduled_departure_date'],
                          ascending=False, inplace=True)
        group['scheduled_rotation_time'] = round((group['scheduled_departure_date'] -
                                                  group['scheduled_arrival_date'].shift(-1)).dt.total_seconds()/60)
        return group
    df = df.groupby(df['tail_number']).apply(compute_rotation_time)
    df['scheduled_rotation_time'] = df['scheduled_rotation_time'].fillna(0)
    return df


def create_previous_delay_code(df):
    def compute_prev_delay_code(group):
        group.sort_values(by=['scheduled_departure_date'],
                          ascending=False, inplace=True)
        group['prev_delay_code'] = group['delay_code'].shift(-1)
        return group
    df = df.groupby(df['tail_number']).apply(compute_prev_delay_code)
    df['prev_delay_code'] = df['prev_delay_code'].fillna('no_delay')
    return df
