import streamlit as st
import pandas as pd
import pickle
import webbrowser

model_standart = pickle.load(open('model_standart.pkl', 'rb'))
model_min_max = pickle.load(open('model_min_max.pkl', 'rb'))
scaler_standart = pickle.load(open('scaler_standart.pkl', 'rb'))
scaler_min_max = pickle.load(open('scaler_min_max.pkl', 'rb'))
url = 'https://www.kaggle.com/code/itaygroer/iris-classification'

st.title('Iris flowers classification App')

st.text("""
This app detect, using machine learning with python.
numpy, pandas, seaborn, matplotlib, scikit-learn, streamlit
""")

st.text("""
On the sidebar you can select the parameters 
to predict the price of the diamond.
""")

st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")

if st.button('Source code', help='Click to open the notebook in kaggle'):
    # webbrowser.open_new_tab(url, autoraise=True)
    webbrowser.open(url, new=2, autoraise=True)


st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')

st.write(df)

st.text("""
The categorical variables are encoded using One Hot Encoding.
The numeric values need to be scaled.
""")

X_s = scaler_standart.transform(df)
X_m = scaler_min_max.transform(df)

X_s = pd.DataFrame(X_s, columns=df.columns)
X_m = pd.DataFrame(X_m, columns=df.columns)

st.subheader('Data after Min Max Scaler')

st.write(X_s)

st.subheader('Data after Standart Scaler')

st.write(X_m)

predict = st.button('Predict!')

dict_res = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

if predict:
    st.write("## Predictions")
    st.code("""
    model.predict(df)
    """, language='python')
    st.write("### Predictions using Standart Scaler")
    res = model_standart.predict(X_s)
    st.write(dict_res[res.argmax()])

    st.write("### Predictions using Min Max Scaler")
    res = model_min_max.predict(X_m)
    st.write(dict_res[res.argmax()])


