import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"D:/hotstar.csv")


print(df.head(5))
print(df.info())

df.set_index('hotstar_id', inplace=True)

print(df.head(5))
print(df.shape)
print(df.describe())


print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=True)


df.drop(columns=['seasons', 'episodes'], inplace=True)

print(df.head(5))


df1 = df.dropna(axis=0, inplace=False, how='any')
print(df1.isnull().sum())


dropped_age_rating = [
    'English', 'Kannada', 'Star Sports 1 Marathi', 'Hindi', 'Tamil',
    'Star Sports 2', 'Star Sports Hindi 1', 'Star Vijay', 'Star Suvarna',
    'Marathi', 'Star Sports 1 Telugu', 'Telugu', 'Star Sports Kannada 1'
]
df = df[~df['age_rating'].isin(dropped_age_rating)]


df['age_rating'] = df['age_rating'].str.replace('U/A 13+', '13')
df['age_rating'] = df['age_rating'].str.replace('U/A 7+', '7')
df['age_rating'] = df['age_rating'].str.replace('U/A 16+', '16')

print(df.age_rating.value_counts())


print(df.genre.value_counts())
print(df.type.value_counts())
print(df.year.value_counts())


df['year'] = df['year'].astype('Int64')
print(df.dtypes)

print(df['year'].head(5))
print(f"Movies are from year {np.min(df.year)} to {np.max(df.year)}")


plt.figure(figsize=(12, 4))
plt.title('Number of shows increases per year', fontdict={'fontsize': 25}, fontweight="bold")
plt.xlabel('Year', fontdict={'fontsize': 15})
plt.ylabel('Count', fontdict={'fontsize': 15})
sns.lineplot(x=df.year.value_counts().index, y=df.year.value_counts(), color='r')


plt.figure(figsize=(12, 4))
plt.rcParams['font.size'] = 14
sns.set_style('darkgrid')
plt.title('Number of shows over the year', fontdict={'fontsize': 25}, fontweight="bold")
plt.xlabel('Year', fontdict={'fontsize': 15})
plt.ylabel('Number of Movie/TV shows', fontdict={'fontsize': 15})
plt.xticks(rotation=90)
sns.countplot(x='year', data=df[df.year > 2005], palette='viridis', hue='year', legend=False)


plt.figure(figsize=(16, 8))
sns.set_style('darkgrid')
plt.title("Top 10 genres present in Hotstar", fontdict={'fontsize': 25}, fontweight='bold')
plt.xlabel('Genre', fontdict={'fontsize': 15})
plt.xticks(rotation=75)
plt.ylabel('Count', fontdict={'fontsize': 15})
data = df.genre.value_counts()[:10]
sns.barplot(x=data.index, y=data)


print(f"Running time is distributed between {np.min(df.running_time)} to {np.max(df.running_time)} minutes")
sns.boxplot(x='running_time', data=df)


Q1 = df.running_time.quantile(0.25)
Q3 = df.running_time.quantile(0.75)
iqr = Q3 - Q1
d = df[(df.running_time > Q3 + (1.5 * iqr)) | (df.running_time < Q1 - (1.5 * iqr))]
sns.boxplot(x='running_time', data=d)


plt.figure(figsize=(16, 8))
sns.histplot(d.running_time, kde=True)


data = df.age_rating.value_counts()
plt.figure(figsize=(16, 8))
sns.set_style('darkgrid')
plt.title('Age Rating of Hotstar Shows', fontdict={'fontsize': 25}, fontweight="bold")
plt.xlabel('Age Rating', fontdict={'fontsize': 15})
plt.xticks(rotation=0)
plt.ylabel('Count', fontdict={'fontsize': 15})
sns.barplot(x=data.index, y=data)


data = df.type.value_counts()
plt.figure(figsize=(8, 8))
plt.pie(data, labels=data.index, shadow=True, startangle=90, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
plt.show()