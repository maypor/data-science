# A program organizes data and displays it in various graphs

import math
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('clubmed_HW2.csv')
print(df.describe())


# Q1a
plt.hist(df.age, histtype='barstacked', label="age_hist")
plt.xlabel("age")
plt.ylabel("frequency")
plt.title("Histogram of client age")
plt.show()

# Q1b
plt.xlabel("age")
plt.ylabel("frequency")
plt.title("Histogram of client age")
w = 3
n = math.ceil((df.age.max() - df.age.min())/w)
ax = plt.hist(df.age, bins=n)
plt.show()

plt.xlabel("age")
plt.ylabel("frequency")
plt.title("Histogram of client age")
w = 15
n = math.ceil((df.age.max() - df.age.min())/w)
ax = plt.hist(df.age, bins=n)
plt.show()

plt.xlabel("age")
plt.ylabel("frequency")
plt.title("Histogram of client age")
w = 29
n = math.ceil((df.age.max() - df.age.min())/w)
ax = plt.hist(df.age, bins=n)
plt.show()

# In the first histogram it can be seen that there is a large and abnormal amount at age 35.
# In the second histogram when the bin is wider, the abnormal amount affects all the values that are in the same bin.
# The greater the width of the bons, the broader, comprehensive and clear picture can be seen.

# Q2
df.club_member.value_counts().plot(kind="bar", title="club members", xlabel="member value", ylabel="count",color="green")

plt.title("club member")
plt.ylabel("count")
plt.yticks(range(0, 100, 10))
plt.show()

# Q3
plt.hist(df.nights, histtype='barstacked', label="age_hist")
plt.xlabel("nights")
plt.title("Night column before log transformation")
plt.show()

# Before we use log transformation the histogram had a right tail .This  means that the Scattering was uneven .
# After  we use log transformation the histogram become mor symmetric .
df["logNights"]=df["nights"].mask(df["nights"]<1,1)
df["logNights"]=np.log10(df["logNights"])
plt.xlabel("nights")
plt.title("Night column after log transformation")
plt.hist(df.logNights)
plt.show()

# Q4a
status_sex_crosstab = pd.crosstab(df["status"], df["sex"], normalize="index")


# Q4b
sex_status_crosstab = pd.crosstab(df["sex"], df["status"], normalize="index")

# Q4c
status_sex_crosstab.plot(kind='bar', stacked=True, title="Proportion table for the distribution of status by sex")
sex_status_crosstab.plot(kind='bar', stacked=True, title="Proportion table for the distribution of sex by status")
plt.show()

# Highest percentage of man in status couple. sometimes the percentage not represent the amount of the subjects .
# In this case the Highest percentage of man in status couple is also the most common status among men.
# The most common status among woman is couple.
# The percentage of married woman is 65%
# the percentage of man in status single is about 50%.

# Q4d
region_accomodation = pd.crosstab(df["region"], df["accomodation"], normalize="index")
region_accomodation.plot(kind='bar', stacked=True, title="Proportion table for the distribution of region by accomodation")
plt.show()

# The conclusion from the graph:
# The preferences of people for rooms or their budget that they can ford for a room according their living area.

# Q4e
sex_status_crosstab = pd.crosstab(df["sex"], df["status"], normalize="index")
sex_status_crosstab .plot(kind='bar', stacked=True)
sex_club_crosstab = pd.crosstab(df["sex"], df["club_member"], normalize="index")
sex_club_crosstab .plot(kind='bar', stacked=True)
plt.show()

# There is a more pronounced relationship between a sex variable and a club member, more men are club members and less women.
# In the relationship between sex and status there is no significant difference between the sexes and their relationships.

# Q5
plt.scatter(df.age, df.minibar)
plt.title("scatter distribution of age and minibar")
plt.xlabel("age")
plt.ylabel("minibar")
plt.show()

# Q6a
NewRoom_price = df["room_price"].replace(to_replace=np.NAN, value=df.room_price.mean())

X = df["room_price"].quantile(0.25)
Y = df["room_price"].quantile(0.75)
price_std = df["room_price"].std()
print("The interquartile range is:", Y-X)
print("The standard deviation of the variable room price is: ", price_std)


# Q6b
mid = df["room_price"].median()
print()
print("The median of the mid variable is:", mid)

print("The amount of values that less than / equal to the median is: ", df["room_price"][df["room_price"] <= mid].count())
# There are 198 records in the file, of which 100 are small / equal to a median,
# a median is a number that is half smaller than it and half larger than it,
# there are 3 records that are equal to a median and therefore the difference.

# Q6c
mean = df["room_price"].mean()
plt.hist(df.room_price)
plt.axvline(x=mean, color="red")
plt.axvline(x=(mean+price_std), color="green")
plt.axvline(x=(mean-price_std), color="green")
plt.show()

# Q6d
plt.hist(df.room_price)
plt.show()
# Room price is right tail distributions.

# Q6e
df.boxplot(column =["age"], by ="ranking", grid = False)

# Q6e.a
# The IQR that spans the widest age range is of rating 2 .

# Q6e.b
plt.axhline(y=97, color="green")
plt.axhline(y=23, color="green")
plt.show()

# Q6f
df.boxplot(column =["visits5years"], by ="age", grid = False)
# The number of visits of the oldest population at age 99 is between 3-4.3.

# Q6g
df.boxplot(column =["room_price"], by ="age", grid = False)
df.boxplot(column =["visits5years"], by ="age", grid = False)
plt.show()
# The oldest population paid about 150 to 250 per night.
# While the rest of the ages pay between 100 and 300, depending on age, there is no uniformity and the price does not depend on age.

# Q6h
plt.scatter(df.ranking, df.total_expenditure)
plt.show()
# There is no significant trend relationship between the rating of the visitors and their expenses,
# it can be seen in the graph that in each rating they spent a different amount of money - the range of expenses is wide in each rating.


# Q7
df['new_visits2016'] = df['visits2016']
df['new_visits2016'] = df['new_visits2016'].replace(to_replace=[0, 1], value="low_visits")
df['new_visits2016'] = df['new_visits2016'].replace(to_replace=[2, 3], value="height_visits")
df['new_visits2016'] = df['new_visits2016'].replace(to_replace=np.nan, value="after2016")
print(df['new_visits2016'].value_counts())

# Q8a
new_total_expenditure = df['total_expenditure']
df['New_total_expenditure'] = new_total_expenditure
df['New_total_expenditure'] = df['New_total_expenditure'].mask(df['New_total_expenditure'] < 0, np.nan)
df['New_total_expenditure'] = df['New_total_expenditure'].replace(to_replace=np.nan, value=df.New_total_expenditure.mean())
Q1 = df['total_expenditure'].quantile(0.25)
Q2 = df['total_expenditure'].quantile(0.5)
Q3 = df['total_expenditure'].quantile(0.75)
Q4 = df['total_expenditure'].quantile(1)
bins = [0, Q1, Q2, Q3, Q4]
labels = ["Q1", 'Q2', "Q3", 'Q4']
df['total_expenditure_new'] = pd.cut(df['New_total_expenditure'], bins=bins, labels=labels)
print(df.total_expenditure_new.head())
print(df.room_price.describe())

# Q8b
# Replacing the missing values with the median is more effective because we want to save the original distributions.
# Mean is not the best replacement because it is influenced by extreme values.

# Q8c
x_std = df["New_total_expenditure"].std()
x_mean = df["New_total_expenditure"].mean()
bins = [x_mean-3*x_std, x_mean-2*x_std, x_mean-x_std, x_mean, x_mean+x_std, x_mean+2*x_std, x_mean+3*x_std]
labels = ["cat_1", 'cat_2', "cat_3", 'cat_4', "cat_5", "cat_6"]
df['total_expenditure_cat'] = pd.cut(df['New_total_expenditure'], bins=bins, labels=labels)
print(df.total_expenditure_cat.head())


# Q9a
df['minibarz'] = stats.zscore(df['minibar'])
print(df[['minibar', 'minibarz']].head(10))

# Q9b
print('the std without normalize: ', df['minibar'].std())
print('the std with normalize:',  df['minibarz'].std())


# 9c
minibarz_max = df['minibarz'].max()
minibarz_min = df['minibarz'].min()
bins = [minibarz_min, -1, 1, minibarz_max]
labels = ['not_typical', 'typical', 'non_typical']
df['minibarz_typ'] = pd.cut(df['minibarz'], bins=bins, labels=labels)
print('The amount of typical values is:', df['minibarz_typ'].value_counts()['typical'])