# group-44-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
 # supress warnings
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('/content/fairfield_loc.csv',sep='\t',encoding='latin-1')
df.head()
df.drop(['Unnamed: 1','Unnamed: 2'],axis=1,inplace=True)
df.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
  
# Label encoding for string values
df['Reservoir']= le.fit_transform(df['Reservoir'])
  
df['Reservoir'].unique()
#### Checking for null values
df.isna().sum()
df_train, df_test = train_test_split(df, train_size = 0.6, test_size = 0.4)

# rescale the features
sc = StandardScaler()

# Scalerizing all numeric columns
numeric_vars = ['LATITUDE_DEGREES', 'LATITUDE_MINUTES', 'LATITUDE_SECONDS',
       'LONGITUDE_DEGREES', 'LONGITUDE_MINUTES', 'LONGITUDE_SECONDS']
df_train[numeric_vars] = sc.fit_transform(df_train[numeric_vars])
df_test[numeric_vars] = sc.fit_transform(df_test[numeric_vars])
df_test.head()
X_train=df_train
y_train=df_train.pop('Reservoir')

X_test=df_test
y_test=df_test.pop('Reservoir')
linear=LinearRegression()
linear.fit(X_train,y_train)

rf=RFE(linear,n_features_to_select=10)
rf.fit(X_train,y_train)
pred=rf.predict(X_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,pred)
