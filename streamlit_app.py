import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Student Performance", page_icon="🎓", layout="wide")

# Load model and preprocessor
@st.cache_resource
def load_model():
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

@st.cache_data
def load_data():
    df = pd.read_csv('artifacts/raw.csv', encoding='latin-1')
    df.columns.values[0] = 'school'
    df.drop_duplicates(inplace=True)
    df['total_score'] = df['G1'] + df['G2'] + df['G3']
    df['average'] = round(df['total_score'] / 3, 2)
    return df

model, preprocessor = load_model()
df = load_data()

# Sidebar
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🤖 Predict"])

# =====================
# HOME PAGE
# =====================
if page == "🏠 Home":
    st.title("🎓 Student Performance Analysis")
    st.markdown("### Portuguese Student Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Features", len(df.columns))
    col3.metric("Avg Grade", round(df['G3'].mean(), 2))
    col4.metric("Model Accuracy", "94.38%")

    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe())

# =====================
# EDA PAGE
# =====================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Grade Distribution", "Categorical Analysis", "Correlation"])

    with tab1:
        st.subheader("Grade Distribution")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(['G1', 'G2', 'G3']):
            sns.histplot(df[col], bins=20, kde=True, ax=axes[i], color='steelblue')
            axes[i].set_title(f'{col} Distribution')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Gender vs Grades")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(['G1', 'G2', 'G3']):
            sns.boxplot(x='sex', y=col, data=df, ax=axes[i], palette='pastel')
            axes[i].set_title(f'Sex vs {col}')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Categorical Features")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df['sex'].value_counts().plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
            ax.set_title('Gender Distribution')
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            df['internet'].value_counts().plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
            ax.set_title('Internet Access')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            df['higher'].value_counts().plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
            ax.set_title('Higher Education Plans')
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            df['address'].value_counts().plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
            ax.set_title('Urban vs Rural')
            st.pyplot(fig)

        st.subheader("Study Time vs G3")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='studytime', y='G3', data=df, palette='viridis', ax=ax)
        ax.set_title('Study Time vs Final Grade')
        st.pyplot(fig)

    with tab3:
        st.subheader("Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=np.number).columns
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# =====================
# PREDICT PAGE
# =====================
elif page == "🤖 Predict":
    st.title("🤖 Predict Student Grade")
    st.markdown("Fill in the student details to predict final grade (G3)")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("School", ['GP', 'MS'])
        sex = st.selectbox("Gender", ['F', 'M'])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Address", ['U', 'R'])
        famsize = st.selectbox("Family Size", ['GT3', 'LE3'])
        Pstatus = st.selectbox("Parent Status", ['T', 'A'])
        Medu = st.slider("Mother Education (0-4)", 0, 4, 2)
        Fedu = st.slider("Father Education (0-4)", 0, 4, 2)
        Mjob = st.selectbox("Mother Job", ['teacher', 'health', 'services', 'at_home', 'other'])
        Fjob = st.selectbox("Father Job", ['teacher', 'health', 'services', 'at_home', 'other'])
        reason = st.selectbox("Reason", ['home', 'reputation', 'course', 'other'])

    with col2:
        guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
        traveltime = st.slider("Travel Time (1-4)", 1, 4, 1)
        studytime = st.slider("Study Time (1-4)", 1, 4, 2)
        failures = st.slider("Past Failures", 0, 3, 0)
        schoolsup = st.selectbox("School Support", ['yes', 'no'])
        famsup = st.selectbox("Family Support", ['yes', 'no'])
        paid = st.selectbox("Paid Classes", ['yes', 'no'])
        activities = st.selectbox("Activities", ['yes', 'no'])
        nursery = st.selectbox("Nursery", ['yes', 'no'])
        higher = st.selectbox("Higher Education", ['yes', 'no'])
        internet = st.selectbox("Internet", ['yes', 'no'])

    with col3:
        romantic = st.selectbox("Romantic", ['yes', 'no'])
        famrel = st.slider("Family Relations (1-5)", 1, 5, 3)
        freetime = st.slider("Free Time (1-5)", 1, 5, 3)
        goout = st.slider("Go Out (1-5)", 1, 5, 3)
        Dalc = st.slider("Weekday Alcohol (1-5)", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol (1-5)", 1, 5, 1)
        health = st.slider("Health (1-5)", 1, 5, 3)
        absences = st.slider("Absences", 0, 50, 0)
        G1 = st.slider("G1 Score (0-20)", 0, 20, 10)
        G2 = st.slider("G2 Score (0-20)", 0, 20, 10)

    if st.button("🎯 Predict Grade", use_container_width=True):
        input_data = pd.DataFrame({
            'school': [school], 'sex': [sex], 'age': [age],
            'address': [address], 'famsize': [famsize], 'Pstatus': [Pstatus],
            'Medu': [Medu], 'Fedu': [Fedu], 'Mjob': [Mjob], 'Fjob': [Fjob],
            'reason': [reason], 'guardian': [guardian], 'traveltime': [traveltime],
            'studytime': [studytime], 'failures': [failures], 'schoolsup': [schoolsup],
            'famsup': [famsup], 'paid': [paid], 'activities': [activities],
            'nursery': [nursery], 'higher': [higher], 'internet': [internet],
            'romantic': [romantic], 'famrel': [famrel], 'freetime': [freetime],
            'goout': [goout], 'Dalc': [Dalc], 'Walc': [Walc],
            'health': [health], 'absences': [absences], 'G1': [G1], 'G2': [G2]
        })

        # Preprocessor se transform karo phir predict karo
        input_transformed = preprocessor.transform(input_data)
        prediction = model.predict(input_transformed)[0]
        prediction = round(max(0, min(20, prediction)))

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted G3", f"{prediction}/20")
        col2.metric("Percentage", f"{prediction*5}%")

        if prediction >= 15:
            col3.metric("Grade", "Excellent! 🌟")
        elif prediction >= 10:
            col3.metric("Grade", "Good! 👍")
        else:
            col3.metric("Grade", "Needs Improvement 📚")