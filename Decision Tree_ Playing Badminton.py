import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
 
# Load historical data from csv file
df = pd.read_csv("historical.csv")

# Change string values into numerical values in Play Badminton
d = {'No': 0, 'Yes': 1}
df['Play Badminton'] = df['Play Badminton'].map(d)

# Drop and make the 'Play Badminton' as target
inputs = df.drop('Play Badminton', axis = 'columns')
target = df['Play Badminton']

in_outlook = LabelEncoder()
in_temperature = LabelEncoder()
in_humidity = LabelEncoder()
in_wind = LabelEncoder()

inputs['Outlook_n'] = in_outlook.fit_transform(inputs['Outlook'])
inputs['Temperature_n'] = in_temperature.fit_transform(inputs['Temperature'])
inputs['Humidity_n'] = in_humidity.fit_transform(inputs['Humidity'])
inputs['Wind_n'] = in_wind.fit_transform(inputs['Wind'])


inputs_n = inputs.drop(['Outlook', 'Temperature', 'Humidity', 'Wind'], axis = 'columns')

features = ['Outlook_n', 'Temperature_n', 'Humidity_n', 'Wind_n']

model = DecisionTreeClassifier()
model = model.fit(inputs_n, target)
data = tree.export_graphviz(model, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('Decision Tree.png')

img=pltimg.imread('Decision Tree.png')
imgplot = plt.imshow(img)
#plt.show()

print(dtree.predict([[40, 10, 6, 1]]))