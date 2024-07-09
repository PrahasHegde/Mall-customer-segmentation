import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans



df = pd.read_csv('Mall_Customers.csv')

print(df.head())
print(df.shape)
print(df.info())

#spending score based on gender and barplot
spending_by_gender = df.groupby('Genre')['Spending Score (1-100)'].mean()
print(spending_by_gender)

plt.bar(spending_by_gender.index,spending_by_gender.values)
plt.show()


#let’s try a scatter plot using the age column on the x-axis and the spending score column on the y-axis.
sns.scatterplot(df, x='Age', y='Spending Score (1-100)')
plt.show()


#annual income on the x-axis and the spending scores on the y-axis.
sns.scatterplot(df, x='Annual Income (k$)', y='Spending Score (1-100)')
plt.show()

"""In the above graph, we can find some clusters. There are some visible groups.
So, these are the two features(Annual Income and Spending Score)we are going to use to do our clustering"""

#let’s create a variable X and store the two features(Annual Income and Spending Score).
X = df.drop(columns=['CustomerID', 'Genre', 'Age']).values
print(X)


"""Inertia is just the distance of all the points from its centroid.If the inertia is low then good if the inertia is high then bad 
When we are using KMeans to cluster features into 3 the KMeans try to find 3 centroids and the points near those centroids are clustered together"""

#model KMeans with Elbow mathod(to determine no. of clusters)
inertia = []

for i in range(2, 11):
  model = KMeans(n_clusters=i, random_state=7)
  model.fit(X)
  inertia.append(model.inertia_)

print(inertia) #as the number of clusters increases the inertia keeps getting low

#ploting graph showing Elbow method (5 is the elbow)
plt.plot(range(2, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

#model using 5 clusters
KMeansModel = KMeans(n_clusters=5, random_state=777)
KMeansModel.fit(X)

#centroids
#The KMeansModel.cluster_centers_ gives us the centers and KMeansModel.predict(X) gives us the labels by which the features are clustered
centers = KMeansModel.cluster_centers_
print(centers)
print(KMeansModel.predict(X)) #predicted labels

center_x = centers[:, 0]
center_y = centers[:, 1]

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=KMeansModel.predict(X))
plt.plot(center_x, center_y, 'xb')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()