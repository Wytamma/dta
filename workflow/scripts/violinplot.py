# seaborn violinplot

def main(log_file):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict

    df = pd.read_csv(log_file, sep='\t', comment='#')

    # melt the dataframe c_Location
    columns = defaultdict(list)

    for column in df.columns:
        if column.startswith('location.root'):
            columns["location.root"].append(column)
        elif column.startswith('location.rates'):
            columns["location.rates"].append(column)
        elif column.startswith('c_Into') or column.startswith('c_OutOf'):
            columns["In.Out"].append(column)
        elif column.startswith('c_Location') and not column.startswith('c_Location.total'):
            columns["Location"].append(column)

    for column in columns:
        print(columns[column])
        melted = pd.melt(df, id_vars=['state'], value_vars=columns[column])
        f, ax = plt.subplots()
        sns.violinplot(x='value', y='variable', data=melted, ax=ax, hue='variable')
        ax.set_title(column)
        ax.set_ylabel(None)
        plt.tight_layout()
        plt.savefig(f'{log_file}_{column}.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args()
    main(args.log_file)